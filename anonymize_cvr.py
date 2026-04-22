#!/usr/bin/env python3
"""
Anonymize Cast Vote Records (CVR) per Colorado C.R.S. 24-72-205.5.

Ballot styles with fewer than MIN_BALLOTS ballots must be aggregated to
protect voter privacy.

Terminology used throughout this module:

  style         — the contest pattern for a ballot, represented as a string
                  of '1' and '0' characters.  Each position corresponds to
                  one contest (in the order contests appear in the CVR file);
                  '1' means the contest is present on the ballot, '0' means
                  it is absent.  This is the only style definition that
                  drives redaction decisions.

  named_style   — the value in the column identified by --stylecol, if any.
                  Assigned by the voting machine; almost certainly obsolete
                  in Colorado, but still accepted.

  ballot_type   — the value in the BallotType column, if present.
"""

import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from cvr_utils import TempCVRFile

MIN_BALLOTS_DEFAULT = 10
NEAR_UNANIMOUS_THRESHOLD = 2  # "all but N votes" triggers balancing (Rule c)
MIN_CONTRASTING_VOTES = 3  # contrasting votes needed per contest after balancing
COVERAGE_WEIGHT = 10.0  # weight for contest coverage vs. vote-balance score


# ---------------------------------------------------------------------------
# CvrDatabase
# ---------------------------------------------------------------------------


class CvrDatabase:
    """
    Reads a CVR file and organizes its contents for analysis and redaction.

    The Colorado CVR CSV format has four header rows, then one ballot per row:
      Row 1: version / election name
      Row 2: contest names (one name per choice column, repeated)
      Row 3: choice (candidate) names, one per column
      Row 4: column headers (CvrNumber, TabulatorNum, BatchId, RecordId,
              ImprintedId, CountingGroup, PrecinctPortion, BallotType, ...)
      Rows 5+: one ballot per row

    Assumption: column 0 of every ballot row is the CvrNumber.  This is
    used throughout the redaction logic to deduplicate ballots and to
    identify which ballots have been aggregated.  No attempt is made to
    locate CvrNumber by name in the header row.

    A ballot's style is a string of '1' and '0' characters, one per contest
    in contest_names order.  '1' means the contest is present on the ballot
    (at least one choice column is non-empty), '0' means it is absent.

    After construction the following are available:

      ballots               — all ballot rows as lists of strings
      contest_names         — ordered list of unique contest names
      contest_to_columns    — contest name -> list of column indices
      contest_choice_meta   — contest name -> {col_idx: choice_name}
      ballots_by_style      — ballots grouped by style string
      ballots_by_named_style — ballots grouped by named_style value
                               (only populated when named_style_col is set)
      ballots_by_ballot_type — ballots grouped by BallotType value
                               (only populated when the column exists)
      ballots_by_style_precinct — ballots grouped by (style, PrecinctPortion) pair
                               (only populated when redact_on_precinct is True
                               and the PrecinctPortion column exists)

    For leakage detection:
      named_styles_by_style — style -> set of named_style values for that style
      ballot_types_by_style — style -> set of ballot_type values for that style
    """

    def __init__(
        self,
        input_file: str,
        headerlen: int,
        named_style_col: Optional[int],
        redact_on_precinct: bool,
    ) -> None:
        """
        Read and validate the CVR file, then build all groupings.

        Args:
            input_file:         Path to the CVR file (CSV or Parquet).
            headerlen:          Number of header columns before vote data begins.
                                Pass 0 to auto-detect from the contests row.
            named_style_col:    Column index of the named_style field, or None.
            redact_on_precinct: If True, populate ballots_by_style_precinct for
                                (style, precinct) combination rare-pair detection.
        """
        self.input_file = input_file
        self.headerlen = (
            headerlen  # may be updated to auto-detected value in _read_file()
        )
        self.named_style_col = named_style_col
        self.redact_on_precinct = redact_on_precinct

        # Four header rows read from the file.
        self.version: List[str] = []
        self.contests: List[str] = []
        self.choices: List[str] = []
        self.headers: List[str] = []

        # All ballot rows.
        self.ballots: List[List[str]] = []

        # Line terminator found in the file (needed when writing output later).
        self.lineterminator: str = ""

        # Column indices for named special columns.  None means not present.
        self.ballot_type_idx: Optional[int] = None
        self.precinct_portion_idx: Optional[int] = None
        self.counting_group_idx: Optional[int] = None

        # Ordered list of unique contest names (defines positions in style strings).
        self.contest_names: List[str] = []

        # contest name -> list of column indices for that contest's choices
        self.contest_to_columns: Dict[str, List[int]] = {}

        # contest name -> {col_idx: choice_name} (used for vote tallying)
        self.contest_choice_meta: Dict[str, Dict[int, str]] = {}

        # Ballots grouped by style string.
        self.ballots_by_style: Dict[str, List[List[str]]] = {}

        # Ballots grouped by named_style value (only when named_style_col is set).
        self.ballots_by_named_style: Dict[str, List[List[str]]] = {}

        # Ballots grouped by BallotType value (only when ballot_type_idx is set).
        self.ballots_by_ballot_type: Dict[str, List[List[str]]] = {}

        # Ballots grouped by (style, PrecinctPortion) pair.
        # Only populated when redact_on_precinct is True and precinct_portion_idx is set.
        self.ballots_by_style_precinct: Dict[Tuple[str, str], List[List[str]]] = {}

        # For leakage detection: per style, which named_style / ballot_type values appear.
        self.named_styles_by_style: Dict[str, Set[str]] = {}
        self.ballot_types_by_style: Dict[str, Set[str]] = {}

        self._read_file()
        self._validate()
        self._build_contest_map()
        self._validate_ballot_contents()
        self._group_ballots()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_file(self) -> None:
        """Read all rows from the CVR file and detect the line terminator."""
        with TempCVRFile(self.input_file) as csv_file:
            # Detect the line terminator so we can reproduce it in the output.
            with open(csv_file, "rb") as raw:
                chunk = raw.read(1024)
                if b"\r\n" in chunk:
                    self.lineterminator = "\r\n"
                elif b"\r" in chunk:
                    self.lineterminator = "\r"
                else:
                    self.lineterminator = "\n"

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                self.version = next(reader)
                self.contests = next(reader)
                self.choices = next(reader)
                self.headers = next(reader)
                self.ballots = []
                for row in reader:
                    if row:
                        self.ballots.append(row)

        # Compute headerlen from the contests row if the caller did not specify one.
        # The contests row begins with one empty cell per header column, then
        # the first non-empty cell is where contest (vote) columns begin.
        if self.headerlen == 0:
            for cell in self.contests:
                if cell.strip() == "":
                    self.headerlen += 1
                else:
                    break
            if self.headerlen == 0:
                raise ValueError(
                    "Could not auto-detect headerlen: no empty cells at the "
                    "start of the contests row."
                )

        # Locate the special columns by name (case-insensitive).
        for idx, name in enumerate(self.headers):
            lower = name.strip().lower()
            if lower == "ballottype":
                self.ballot_type_idx = idx
            elif lower == "precinctportion":
                self.precinct_portion_idx = idx
            elif lower == "countinggroup":
                self.counting_group_idx = idx

    def _validate(self) -> None:
        """Check that the file structure is usable."""
        if self.headerlen >= len(self.contests):
            raise ValueError(
                f"No vote columns found: headerlen={self.headerlen} but "
                f"contests row has only {len(self.contests)} columns."
            )
        if self.named_style_col is not None and self.named_style_col >= self.headerlen:
            raise ValueError(
                f"Named style column {self.named_style_col} must be within "
                f"the header columns (headerlen={self.headerlen})."
            )
        # Every ballot row must be exactly as wide as the contests row.
        expected_cols = len(self.contests)
        for i, ballot in enumerate(self.ballots):
            if len(ballot) != expected_cols:
                raise ValueError(
                    f"Row {i + 5} has {len(ballot)} columns but contests row "
                    f"has {expected_cols}."
                )

    def _build_contest_map(self) -> None:
        """Build the contest name -> column indices mapping and ordered contest list."""
        mapping: Dict[str, List[int]] = defaultdict(list)
        for col_idx in range(self.headerlen, len(self.contests)):
            name = self.contests[col_idx].strip()
            if not name:
                raise ValueError(
                    f"Contest row column {col_idx} is empty; expected a contest name."
                )
            if name not in mapping:
                self.contest_names.append(name)
            mapping[name].append(col_idx)
        self.contest_to_columns = dict(mapping)

        # Build choice metadata: for each contest, map column index to choice name.
        self.contest_choice_meta = {}
        for contest_name, col_indices in self.contest_to_columns.items():
            col_map: Dict[int, str] = {}
            for col_idx in col_indices:
                choice_name = (
                    self.choices[col_idx].strip() if col_idx < len(self.choices) else ""
                )
                col_map[col_idx] = choice_name if choice_name else f"Choice{col_idx}"
            self.contest_choice_meta[contest_name] = col_map

    def _should_skip_ballot(self, ballot: List[str]) -> bool:
        """
        Return True if this row is an artifact of a previous redaction pass
        and should be excluded from style groupings and content validation.

        Skips two kinds of rows:
          - Redacted ballot rows: any vote column contains '*'
          - The aggregate summary row: BallotType == "AGGREGATED" or CvrNumber == "AGGREGATED"
        """
        for col_idx in range(self.headerlen, len(ballot)):
            if ballot[col_idx].strip() == "*":
                return True
        if self.ballot_type_idx is not None:
            if ballot[self.ballot_type_idx].strip() == "AGGREGATED":
                return True
        if ballot[0].strip() == "AGGREGATED":
            return True
        return False

    def _validate_ballot_contents(self) -> None:
        """
        Verify that for each contest on each ballot, choice columns are either
        all empty (contest absent) or all non-empty (contest present).
        Mixed state indicates a malformed ballot row.
        """
        for i, ballot in enumerate(self.ballots):
            if self._should_skip_ballot(ballot):
                continue
            for contest_name, col_indices in self.contest_to_columns.items():
                values = [ballot[col_idx].strip() for col_idx in col_indices]
                empty_flags = [v == "" for v in values]
                if any(empty_flags) and not all(empty_flags):
                    raise ValueError(
                        f"Row {i + 5}: contest '{contest_name}' has mixed "
                        f"empty/non-empty choice columns."
                    )

    def _style_for_ballot(self, ballot: List[str]) -> str:
        """
        Compute the style string for one ballot.

        Returns a string of '1' and '0' characters, one per contest in
        contest_names order.  '1' means the contest is present on the ballot.
        """
        parts = []
        for contest_name in self.contest_names:
            col_indices = self.contest_to_columns[contest_name]
            present = False
            for col_idx in col_indices:
                if ballot[col_idx].strip() != "":
                    present = True
                    break
            parts.append("1" if present else "0")
        return "".join(parts)

    def _group_ballots(self) -> None:
        """
        Iterate all ballots once and build every grouping needed for
        analysis and redaction.
        """
        by_style: Dict[str, List[List[str]]] = defaultdict(list)
        by_named_style: Dict[str, List[List[str]]] = defaultdict(list)
        by_ballot_type: Dict[str, List[List[str]]] = defaultdict(list)
        by_style_precinct: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)
        named_styles_by_style: Dict[str, Set[str]] = defaultdict(set)
        ballot_types_by_style: Dict[str, Set[str]] = defaultdict(set)

        for ballot in self.ballots:
            if self._should_skip_ballot(ballot):
                continue
            style = self._style_for_ballot(ballot)
            by_style[style].append(ballot)

            if self.named_style_col is not None:
                named_style = ballot[self.named_style_col].strip()
                by_named_style[named_style].append(ballot)
                named_styles_by_style[style].add(named_style)

            if self.ballot_type_idx is not None:
                ballot_type = ballot[self.ballot_type_idx].strip()
                by_ballot_type[ballot_type].append(ballot)
                if ballot_type:
                    ballot_types_by_style[style].add(ballot_type)

            if self.redact_on_precinct and self.precinct_portion_idx is not None:
                precinct = ballot[self.precinct_portion_idx].strip()
                by_style_precinct[(style, precinct)].append(ballot)

        self.ballots_by_style = dict(by_style)
        self.ballots_by_named_style = dict(by_named_style)
        self.ballots_by_ballot_type = dict(by_ballot_type)
        self.ballots_by_style_precinct = dict(by_style_precinct)
        self.named_styles_by_style = dict(named_styles_by_style)
        self.ballot_types_by_style = dict(ballot_types_by_style)


# ---------------------------------------------------------------------------
# RedactionNeeds
# ---------------------------------------------------------------------------


class RedactionNeeds:
    """
    Describes what the CVR requires before it can be safely published.

    Populated by check_redaction_needs().  The redaction logic reads this to
    decide what work to do.

    Rare styles and rare (style, precinct) pairs both require the same kind of
    treatment: ballots must be aggregated so no individual voter can be identified.
    """

    def __init__(self) -> None:
        # Styles with too few ballots.
        # Key: style string.  Value: ballot count.
        self.rare_styles: Dict[str, int] = {}

        # (style, precinct) pairs with too few ballots.
        # Only populated when --redact-on-precinct is requested.
        # Per C.R.S. 24-72-205.5, the privacy unit is the combination of contest
        # pattern and precinct, not each independently.
        # Key: (style string, PrecinctPortion value).  Value: ballot count.
        self.rare_style_precinct_pairs: Dict[Tuple[str, str], int] = {}

        # Human-readable leakage warnings.  Leakage is reported but not corrected.
        self.leakage_warnings: List[str] = []

    def needs_redaction(self) -> bool:
        """Return True if any redaction work is required."""
        return len(self.rare_styles) > 0 or len(self.rare_style_precinct_pairs) > 0


# ---------------------------------------------------------------------------
# check_redaction_needs
# ---------------------------------------------------------------------------


def check_redaction_needs(
    db: CvrDatabase,
    min_ballots: int,
) -> RedactionNeeds:
    """
    Examine the database and return a description of what needs redacting.

    Args:
        db:          The CVR database to examine.
        min_ballots: Minimum ballots required per style or precinct.

    Returns:
        A RedactionNeeds object describing what must be done.
    """
    needs = RedactionNeeds()

    # Check for rare styles (Rule 2).
    for style, ballots in db.ballots_by_style.items():
        if len(ballots) < min_ballots:
            needs.rare_styles[style] = len(ballots)

    # Check for rare (style, precinct) combinations.
    # ballots_by_style_precinct is only populated when --redact-on-precinct was set.
    # Per statute, the privacy unit is the combination of contest pattern and precinct,
    # so a pair is rare when it has fewer than min_ballots ballots even if both the
    # style and the precinct individually have enough ballots.
    for pair, ballots in db.ballots_by_style_precinct.items():
        if len(ballots) < min_ballots:
            needs.rare_style_precinct_pairs[pair] = len(ballots)

    # Check for named_style leakage (Rule 10).
    # Leakage: more than one named_style maps to the same contest pattern.
    if db.named_style_col is not None:
        for style, named_styles in db.named_styles_by_style.items():
            if len(named_styles) > 1:
                names = ", ".join(sorted(named_styles))
                needs.leakage_warnings.append(
                    f"Leakage: named styles [{names}] all share the same contest pattern"
                )

    # Check for ballot_type leakage (Rule 10).
    # Leakage: more than one ballot_type maps to the same contest pattern.
    if db.ballot_type_idx is not None:
        for style, ballot_types in db.ballot_types_by_style.items():
            if len(ballot_types) > 1:
                types = ", ".join(sorted(ballot_types))
                needs.leakage_warnings.append(
                    f"Leakage: ballot types [{types}] all share the same contest pattern"
                )

    return needs


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class Aggregate:
    """
    Accumulates ballots into the anonymized aggregate pool, tracking
    counts needed to verify the redaction rules.

    All four anonymization rules pertain to building the aggregate:

    Rule a: The aggregate must contain at least min_ballots ballots in total.

    Rule b: For each rare contest (one that appears on at least one rare
            ballot), the aggregate must contain at least min_ballots ballots
            that include that contest.

    Rule c: No contest in the aggregate may be near-unanimous.  "Near-
            unanimous" means all but NEAR_UNANIMOUS_THRESHOLD (default
            value = 2) votes go to a single choice.  If a contest is
            near-unanimous, contrasting ballots are borrowed from the
            common pool until at least MIN_CONTRASTING_VOTES (default
            value = 3) ballots vote for a non-leading choice. Only contests
            on rare ballots are checked; near-unanimity in contests that
            belong only to common styles is not a concern.

    Rule d: A ballot may only be borrowed from a common style if that style
            will still have at least min_ballots ballots remaining after the
            borrow.  Enforced by CommonPool, which removes a style from the
            pool entirely once it would drop below the minimum.
    """

    def __init__(
        self,
        initial_ballots: List[List[str]],
        rare_contests: Set[str],
        db: CvrDatabase,
        min_ballots: int,
    ) -> None:
        self._db = db
        self.min_ballots = min_ballots
        self.rare_contests = rare_contests
        self.ballots: List[List[str]] = []
        self._cvr_numbers: Set[str] = set()

        # For each contest, how many ballots in the aggregate include that contest.
        self._contest_ballot_counts: Dict[str, int] = defaultdict(int)

        # For each contest (the list of contests already exists in the db and will
        # not change), establish a dictionary which maps the contest name to an inner
        # dictionary (which will be filled in by add()) which will map the contest's
        # choices to the number of votes for each choice.
        self._contest_choice_counts: Dict[str, Dict[str, int]] = {
            c: {} for c in db.contest_to_columns
        }
        for ballot in initial_ballots:
            self.add(ballot)

    def add(self, ballot: List[str]) -> None:
        """Add a ballot to the aggregate, updating all tracked counts."""
        self.ballots.append(ballot)
        cvr_num = ballot[0].strip()
        if cvr_num:
            self._cvr_numbers.add(cvr_num)

        for contest_name, col_indices in self._db.contest_to_columns.items():
            present = any(ballot[col_idx].strip() != "" for col_idx in col_indices)
            if present:
                self._contest_ballot_counts[contest_name] += 1

        for contest_name, col_map in self._db.contest_choice_meta.items():
            for col_idx, choice_name in col_map.items():
                val = ballot[col_idx].strip()
                if not val or val == "0":
                    continue
                try:
                    increment = int(float(val))
                except ValueError:
                    increment = 1
                # get the inner [choice_name:count] dict for the contest.
                counts = self._contest_choice_counts[contest_name]
                counts[choice_name] = counts.get(choice_name, 0) + increment

    def contains_cvr(self, cvr_num: str) -> bool:
        return cvr_num in self._cvr_numbers

    def total_count(self) -> int:
        return len(self.ballots)

    def needs_more_total_ballots(self) -> bool:
        """True if the aggregate does not yet have min_ballots total (Rule a)."""
        return len(self.ballots) < self.min_ballots

    def contests_needing_ballots(self) -> Dict[str, int]:
        """
        Return {contest: ballots_still_needed} for Rule b (each rare contest
        needs >= min_ballots).
        """
        result = {}
        for contest in self.rare_contests:
            count = self._contest_ballot_counts[contest]
            if count < self.min_ballots:
                result[contest] = self.min_ballots - count
        return result

    def satisfies_minimums(self) -> bool:
        """True when both Rule a and Rule b are satisfied."""
        return (
            not self.needs_more_total_ballots()
            and len(self.contests_needing_ballots()) == 0
        )

    def choice_counts(self) -> Dict[str, Dict[str, int]]:
        """Return per-contest per-choice vote counts (used for unanimity checking)."""
        return self._contest_choice_counts


# ---------------------------------------------------------------------------
# CommonPool
# ---------------------------------------------------------------------------


class CommonPool:
    """
    Pool of common-style ballots available for borrowing into the aggregate.

    Enforces Rule d: removing a ballot from a style never leaves that style
    with fewer than min_ballots ballots.  Styles that would fall below the
    minimum are removed from the pool entirely.
    """

    def __init__(self, styles: Dict[str, List[List[str]]], min_ballots: int) -> None:
        self._styles: Dict[str, List[List[str]]] = {
            sig: list(rows) for sig, rows in styles.items()
        }
        self.min_ballots = min_ballots

    def is_empty(self) -> bool:
        return len(self._styles) == 0

    def styles(self) -> Dict[str, List[List[str]]]:
        return self._styles

    def remove(self, style_sig: str, row_idx: int) -> None:
        """Remove one ballot; remove the style from the pool if it drops below min_ballots."""
        rows = self._styles[style_sig]
        rows.pop(row_idx)
        if len(rows) < self.min_ballots:
            del self._styles[style_sig]

    def remove_by_cvr_numbers(self, cvr_numbers: Set[str]) -> None:
        """Remove all ballots whose CvrNumber is in the given set."""
        for style_sig in list(self._styles.keys()):
            remaining = [
                row
                for row in self._styles[style_sig]
                if row[0].strip() not in cvr_numbers
            ]
            if len(remaining) < self.min_ballots:
                del self._styles[style_sig]
            else:
                self._styles[style_sig] = remaining

    def best_candidate_for(
        self, aggregate: Aggregate, db: CvrDatabase
    ) -> Optional[Tuple[str, int, List[str]]]:
        """
        Return (style_sig, row_idx, ballot) for the best ballot to borrow, or None.

        Prioritizes ballots that cover the most contests still needing ballots
        (Rule b), weighted by how much they reduce vote imbalance.  Falls back
        to any ballot from the largest available style when only the total count
        is short (Rule a).
        """
        needed = aggregate.contests_needing_ballots()
        if needed:
            return self._best_for_contests(aggregate, db, needed)
        elif aggregate.needs_more_total_ballots():
            return self._any_candidate(aggregate)
        else:
            return None

    def _best_for_contests(
        self,
        aggregate: Aggregate,
        db: CvrDatabase,
        needed: Dict[str, int],
    ) -> Optional[Tuple[str, int, List[str]]]:
        """
        Find the best ballot for bringing the aggregation into agreement with Rule b
        (has at least min_ballots per contest).

        The arg "needed" is a dictionary: {contest:number of ballots needed for that contest}
        """

        needed_list = [c for c, n in needed.items() if n > 0]
        best_candidate: Optional[Tuple[str, int, List[str]]] = None
        best_score = -1.0

        for style_sig, rows in self._styles.items():
            if len(rows) <= self.min_ballots:
                continue
            for idx, row in enumerate(rows):
                cvr_num = row[0].strip()
                if cvr_num and aggregate.contains_cvr(cvr_num):
                    continue
                covered = [
                    contest
                    for contest in needed_list
                    if _ballot_has_contest(row, contest, db.contest_to_columns)
                ]
                if not covered:
                    continue
                # "gain" indicates an improvement in the score, even though it is
                # accomplished by a "reduction" in the imbalance.
                gain = sum(
                    _imbalance_reduction(
                        c, row, aggregate.choice_counts(), db.contest_choice_meta
                    )
                    for c in covered
                )
                score = COVERAGE_WEIGHT * len(covered) + gain
                if score > best_score:
                    best_score = score
                    best_candidate = (style_sig, idx, row)

        return best_candidate

    def _any_candidate(
        self, aggregate: Aggregate
    ) -> Optional[Tuple[str, int, List[str]]]:
        """Pick any ballot from the largest borrowable style."""
        best_sig = None
        best_count = 0
        for style_sig, rows in self._styles.items():
            if len(rows) > self.min_ballots and len(rows) > best_count:
                best_sig = style_sig
                best_count = len(rows)
        if best_sig is None:
            return None
        for idx, row in enumerate(self._styles[best_sig]):
            cvr_num = row[0].strip()
            if cvr_num and aggregate.contains_cvr(cvr_num):
                continue
            return (best_sig, idx, row)
        return None


# ---------------------------------------------------------------------------
# Redaction helpers
# ---------------------------------------------------------------------------


def _ballot_has_contest(
    ballot: List[str], contest: str, contest_to_columns: Dict[str, List[int]]
) -> bool:
    """Return True if the ballot has any non-empty column for the given contest."""
    col_indices = contest_to_columns.get(contest, [])
    return any(ballot[col_idx].strip() != "" for col_idx in col_indices)


def _imbalance_reduction(
    contest: str,
    ballot: List[str],
    choice_counts: Dict[str, Dict[str, int]],
    contest_choice_meta: Dict[str, Dict[int, str]],
) -> float:
    """
    Estimate how much adding this ballot reduces vote imbalance for a contest.

    Imbalance is max_choice_votes minus the sum of all other votes.
    Returns the improvement (positive = less imbalanced), or 0.0 if the ballot
    does not participate in the contest or makes imbalance worse.
    """

    # "current" is the dict that maps choices to votes for that contest.
    # "total" is the sum of all votes for all choices
    # current_max is the vote total for the choice with the highest number of votes.
    current = choice_counts.get(contest, {})
    current_total = sum(current.values())
    current_max = max(current.values()) if current else 0
    current_gap = current_max - (current_total - current_max)

    # record the votes on this ballot for each of the choices available for this contest
    contributions: Dict[str, int] = {}
    for col_idx, choice_name in contest_choice_meta[contest].items():
        val = ballot[col_idx].strip()
        if not val or val == "0":
            continue
        contributions[choice_name] = 1

    if not contributions:
        return 0.0

    new_counts = dict(current)
    for choice_name, inc in contributions.items():
        new_counts[choice_name] = new_counts.get(choice_name, 0) + inc
    new_total = current_total + sum(contributions.values())
    new_max = max(new_counts.values()) if new_counts else 0
    new_gap = new_max - (new_total - new_max)

    return max(0.0, current_gap - new_gap)


def _build_aggregate_row(
    ballots: List[List[str]], db: CvrDatabase, aggregate_id: str
) -> List[str]:
    """
    Build the AGGREGATED CSV row by summing vote columns and blanking identifiers.

    Fixed header columns (TabulatorNum through ImprintedId) are blanked.
    CountingGroup, PrecinctPortion, and named_style are blanked.
    BallotType is set to "AGGREGATED".
    Vote columns are summed across all ballots in the aggregate.
    """
    if not ballots:
        return []

    result = ballots[0][: db.headerlen].copy()
    result[0] = aggregate_id
    if len(result) > 1:
        result[1] = ""  # TabulatorNum
    if len(result) > 2:
        result[2] = ""  # BatchId
    if len(result) > 3:
        result[3] = ""  # RecordId
    if len(result) > 4:
        result[4] = ""  # ImprintedId
    if db.named_style_col is not None:
        result[db.named_style_col] = ""
    if db.counting_group_idx is not None:
        result[db.counting_group_idx] = ""
    if db.precinct_portion_idx is not None:
        result[db.precinct_portion_idx] = ""
    if db.ballot_type_idx is not None:
        result[db.ballot_type_idx] = "AGGREGATED"

    # Sum vote columns across all ballots.
    num_cols = len(ballots[0])
    for col_idx in range(db.headerlen, num_cols):
        total = 0.0
        for ballot in ballots:
            val = ballot[col_idx].strip()
            if val and val.replace(".", "").replace("-", "").isdigit():
                try:
                    total += float(val)
                except ValueError:
                    pass
        if total == int(total):
            result.append(str(int(total)))
        else:
            result.append(str(total))

    return result


def _redact_ballot_row(ballot: List[str], db: CvrDatabase) -> List[str]:
    """
    Return a copy of the ballot with all vote columns replaced by '*'.

    CountingGroup and PrecinctPortion are always blanked on redacted rows,
    regardless of the --redact-on-precinct setting.
    """
    result = ballot.copy()
    if db.counting_group_idx is not None:
        result[db.counting_group_idx] = ""
    if db.precinct_portion_idx is not None:
        result[db.precinct_portion_idx] = ""
    for col_idx in range(db.headerlen, len(result)):
        result[col_idx] = "*"
    return result


def _blank_geographic_fields(
    ballot: List[str], db: CvrDatabase, redact_on_precinct: bool
) -> List[str]:
    """
    Return a copy of the ballot with geographic fields blanked.

    CountingGroup is always blanked.

    PrecinctPortion handling:
      - If redact_on_precinct is False: blank it on all ballots, because
        precinct information is not safe to publish without per-precinct
        redaction.
      - If redact_on_precinct is True: keep it on non-redacted ballots,
        because rare precincts have already been aggregated and the
        remaining precinct values are safe.
    """
    result = ballot.copy()
    if db.counting_group_idx is not None:
        result[db.counting_group_idx] = ""
    if not redact_on_precinct and db.precinct_portion_idx is not None:
        result[db.precinct_portion_idx] = ""
    return result


def _print_borrowing_needs(aggregate: Aggregate, min_ballots: int) -> None:
    """Print a one-time summary of why ballot borrowing is needed."""
    if aggregate.needs_more_total_ballots():
        needed = min_ballots - aggregate.total_count()
        print(
            f"  Aggregate has {aggregate.total_count()} ballot(s); "
            f"need {needed} more to reach minimum of {min_ballots}."
        )
    contest_needs = aggregate.contests_needing_ballots()
    if contest_needs:
        print("  Contests below minimum:")
        for contest, needed in sorted(contest_needs.items()):
            print(f"    '{contest}': needs {needed} more ballot(s)")


def build_aggregate(
    rare_ballots: List[List[str]],
    rare_contests: Set[str],
    pool: CommonPool,
    db: CvrDatabase,
    min_ballots: int,
) -> Aggregate:
    """
    Build the aggregate from rare ballots, borrowing from the pool as needed.

    Satisfies Rules a and b simultaneously using a coverage-weighted ballot
    selection: each borrowed ballot is chosen to cover as many under-represented
    contests as possible while also reducing vote imbalance.  Rule d (only borrow
    from styles that remain common after borrowing) is enforced by CommonPool.
    """
    aggregate = Aggregate(rare_ballots, rare_contests, db, min_ballots)

    if not aggregate.satisfies_minimums():
        print(f"  Rare ballots: {aggregate.total_count()}.")
        print("  Borrowing from common styles to satisfy these requirements:")
        print(
            f"  - the ballot aggregation must contain at least {min_ballots} ballots."
        )
        print(
            f"  - every contest in the rare styles must appear on at least"
            f" {min_ballots} ballots in the aggregation."
        )
        print()
        _print_borrowing_needs(aggregate, min_ballots)
        print(
            "\n  Selecting ballots to borrow from common styles (this can take a few minutes)..."
        )

    borrowed_cvr_nums: List[str] = []

    while not aggregate.satisfies_minimums():
        result = pool.best_candidate_for(aggregate, db)
        if result is None:
            for contest, needed in aggregate.contests_needing_ballots().items():
                print(
                    f"Warning: could not find enough ballots for contest "
                    f"'{contest}' (still need {needed} more). "
                    f"This contest may only appear on rare ballot styles.",
                    file=sys.stderr,
                )
            break
        style_sig, row_idx, ballot = result
        borrowed_cvr_nums.append(ballot[0].strip())
        aggregate.add(ballot)
        pool.remove(style_sig, row_idx)

    if borrowed_cvr_nums:
        print()
        print(
            f"  Borrowed {len(borrowed_cvr_nums)} ballot(s): {', '.join(borrowed_cvr_nums)}"
        )

    return aggregate


def balance_unanimity(
    aggregate: Aggregate,
    pool: CommonPool,
    db: CvrDatabase,
) -> None:
    """
    Add contrasting ballots to prevent near-unanimous vote patterns (Rule c).

    Only checks contests that appeared on rare ballots; near-unanimity in
    contests that belong only to common styles is not a concern here.
    """
    contest_totals = aggregate.choice_counts()

    # Construct a list of contests that are in the "near-unamimous" condition.
    # That is, find contests whose top vote-holder is within NEAR_UNANIMOUS_THRESHOLD
    # votes of the total votes in the contest.
    problematic: List[Tuple] = []
    for contest_name, choice_votes in contest_totals.items():
        if contest_name not in aggregate.rare_contests:
            continue
        if not choice_votes:
            continue
        total_votes = sum(choice_votes.values())
        if total_votes == 0:
            continue
        max_choice = max(choice_votes, key=lambda c: choice_votes[c])
        max_votes = choice_votes[max_choice]
        other_votes = total_votes - max_votes
        if other_votes <= NEAR_UNANIMOUS_THRESHOLD:
            problematic.append((contest_name, max_choice, max_votes, total_votes))

    if not problematic:
        print("  There are no near-unanimous contests.")
        return

    for contest_name, max_choice, max_votes, total_votes in problematic:
        print(
            f"    '{contest_name}': '{max_choice}' has "
            f"{max_votes} out of {total_votes} votes"
        )

    contrasting = find_contrasting_ballots_multi(problematic, pool, db)
    if contrasting:
        borrowed_cvr_nums = {b[0].strip() for b in contrasting if b[0].strip()}
        pool.remove_by_cvr_numbers(borrowed_cvr_nums)
        for ballot in contrasting:
            aggregate.add(ballot)
        print()
        print(
            f"  Borrowed {len(borrowed_cvr_nums)} contrasting ballot(s): "
            f"{', '.join(sorted(borrowed_cvr_nums))}"
        )


def find_contrasting_ballots_multi(
    problematic_contests: List[Tuple],
    pool: CommonPool,
    db: CvrDatabase,
) -> List[List[str]]:
    """
    Find ballots from the pool that vote differently in near-unanimous contests.

    Minimizes total ballots borrowed by preferring ballots that address multiple
    problematic contests at once.  Aims for MIN_CONTRASTING_VOTES differing
    ballots per contest.
    """
    if not problematic_contests:
        return []

    # For each problematic contest, find which column holds the leading choice.
    contest_info: Dict[str, Dict] = {}
    for contest_name, winning_choice, _, _ in problematic_contests:
        col_indices = db.contest_to_columns.get(contest_name, [])
        if not col_indices:
            continue
        col_map = db.contest_choice_meta.get(contest_name, {})
        winning_col = None
        for col_idx, choice_name in col_map.items():
            if choice_name == winning_choice:
                winning_col = col_idx
                break
        contest_info[contest_name] = {
            "col_indices": col_indices,
            "winning_col": winning_col,
        }

    # Score each available ballot by how many problematic contests it votes against.
    ballot_scores: List[Tuple[int, List[str], List[str]]] = []
    for style_sig, rows in pool.styles().items():
        if len(rows) <= pool.min_ballots:
            continue
        for ballot in rows:
            satisfied = []
            for contest_name, _, _, _ in problematic_contests:
                if contest_name not in contest_info:
                    continue
                info = contest_info[contest_name]
                has_contest = any(
                    ballot[col_idx].strip() != "" for col_idx in info["col_indices"]
                )
                if not has_contest:
                    continue
                winning_col = info["winning_col"]
                if winning_col is not None and ballot[winning_col].strip() != "1":
                    for col_idx in info["col_indices"]:
                        if col_idx != winning_col and ballot[col_idx].strip() == "1":
                            satisfied.append(contest_name)
                            break
            if satisfied:
                ballot_scores.append((len(satisfied), satisfied, ballot))

    ballot_scores.sort(key=lambda x: x[0], reverse=True)

    # Greedily select ballots until each problematic contest has enough contrast.
    contests_needed: Set[str] = {name for name, _, _, _ in problematic_contests}
    selected: List[List[str]] = []
    contrast_counts: Dict[str, int] = defaultdict(int)

    for score, satisfied, ballot in ballot_scores:
        if set(satisfied) & contests_needed:
            selected.append(ballot)
            for c in satisfied:
                contrast_counts[c] += 1
            for c in list(contests_needed):
                if contrast_counts[c] >= MIN_CONTRASTING_VOTES:
                    contests_needed.discard(c)
            if not contests_needed:
                break

    return selected


def _display_contest_name(name: str) -> str:
    """Strip trailing '(Vote For=N)' from a contest name for display."""
    idx = name.find(" (Vote For=")
    return name[:idx] if idx >= 0 else name


def _verify_redaction_tally(
    db: CvrDatabase,
    aggregated_cvr_nums: Set[str],
    aggregate_row: List[str],
) -> None:
    """
    Verify that vote totals are preserved across the full CVR after redaction,
    and print a comparison table.

    Redacted total = (full original total) - (aggregated ballots) + (aggregate row).
    These must equal the full original total, which requires that the aggregate row
    exactly captures all votes from the aggregated ballots.  Raises ValueError if
    a mismatch is found.
    """
    # Tally votes from ALL original ballots.
    full_tally: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ballot in db.ballots:
        for contest_name, col_map in db.contest_choice_meta.items():
            for col_idx, choice_name in col_map.items():
                if ballot[col_idx].strip() == "1":
                    full_tally[contest_name][choice_name] += 1

    # Tally votes from only the aggregated ballots.
    aggregated_tally: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ballot in db.ballots:
        if ballot[0].strip() not in aggregated_cvr_nums:
            continue
        for contest_name, col_map in db.contest_choice_meta.items():
            for col_idx, choice_name in col_map.items():
                if ballot[col_idx].strip() == "1":
                    aggregated_tally[contest_name][choice_name] += 1

    # Tally votes from the aggregate row.
    row_tally: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for contest_name, col_map in db.contest_choice_meta.items():
        for col_idx, choice_name in col_map.items():
            if col_idx >= len(aggregate_row):
                continue
            val = aggregate_row[col_idx].strip()
            if val:
                try:
                    count = int(float(val))
                    if count > 0:
                        row_tally[contest_name][choice_name] += count
                except ValueError:
                    pass

    # Compute the redacted tally: full - aggregated ballots + aggregate row.
    all_contests = set(full_tally.keys()) | set(row_tally.keys())
    redacted_tally: Dict[str, Dict[str, int]] = {}
    for contest_name in all_contests:
        full_c = full_tally.get(contest_name, {})
        agg_c = aggregated_tally.get(contest_name, {})
        row_c = row_tally.get(contest_name, {})
        all_choices = set(full_c.keys()) | set(agg_c.keys()) | set(row_c.keys())
        redacted_tally[contest_name] = {}
        for choice_name in all_choices:
            redacted_tally[contest_name][choice_name] = (
                full_c.get(choice_name, 0)
                - agg_c.get(choice_name, 0)
                + row_c.get(choice_name, 0)
            )

    # Print the full CVR tally comparison table.
    NAME_COL = 40
    print(f"\n  {'Contest/Choice':<{NAME_COL}} {'Original':>10} {'Redacted':>10}")
    print(f"  {'-' * NAME_COL} {'-' * 10} {'-' * 10}")
    for contest_name in sorted(all_contests):
        print(f"  {_display_contest_name(contest_name)}")
        full_c = full_tally.get(contest_name, {})
        red_c = redacted_tally.get(contest_name, {})
        all_choices = set(full_c.keys()) | set(red_c.keys())
        for choice_name in sorted(all_choices):
            o = full_c.get(choice_name, 0)
            r = red_c.get(choice_name, 0)
            if o > 0 or r > 0:
                print(f"    {choice_name:<{NAME_COL - 2}} {o:>10} {r:>10}")

    # Verify: aggregate row must exactly match the aggregated ballots.
    mismatches = []
    for contest_name in sorted(set(aggregated_tally.keys()) | set(row_tally.keys())):
        agg = aggregated_tally.get(contest_name, {})
        row = row_tally.get(contest_name, {})
        all_choices = set(agg.keys()) | set(row.keys())
        for choice_name in sorted(all_choices):
            a = agg.get(choice_name, 0)
            r = row.get(choice_name, 0)
            if a != r:
                mismatches.append((contest_name, choice_name, a, r))

    if mismatches:
        print(
            "ERROR: Tally mismatch between aggregated ballots and aggregate row:",
            file=sys.stderr,
        )
        for contest, choice, expected, got in mismatches:
            print(
                f"  '{contest}' / '{choice}': "
                f"aggregated ballots={expected}, aggregate row={got}",
                file=sys.stderr,
            )
        raise ValueError(
            "Anonymization failed: aggregate row tallies do not match original. "
            "This indicates a bug in the aggregation logic."
        )


def perform_redaction(
    db: CvrDatabase,
    needs: RedactionNeeds,
    min_ballots: int,
    output_file: str,
    redact_on_precinct: bool,
) -> None:
    """
    Perform redaction and write the anonymized CVR to output_file.

    If no redaction is needed, writes the file with only geographic fields
    blanked and no aggregate row appended.

    For each ballot in the aggregate (rare or borrowed), the vote columns are
    replaced with '*' and the row appears in its original position in the file.
    The AGGREGATED row with summed vote counts is appended at the end.
    """
    if not needs.needs_redaction():
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator=db.lineterminator)
            writer.writerow(db.version)
            writer.writerow(db.contests)
            writer.writerow(db.choices)
            writer.writerow(db.headers)
            for ballot in db.ballots:
                writer.writerow(
                    _blank_geographic_fields(ballot, db, redact_on_precinct)
                )
        print("No redaction needed.  Output written.")
        return

    # Collect the initial set of ballots to aggregate:
    # all ballots from rare styles, plus rare precincts if applicable.
    initial_rare_cvr_nums: Set[str] = set()
    rare_ballots: List[List[str]] = []

    for style in needs.rare_styles:
        for ballot in db.ballots_by_style.get(style, []):
            cvr_num = ballot[0].strip()
            if cvr_num not in initial_rare_cvr_nums:
                initial_rare_cvr_nums.add(cvr_num)
                rare_ballots.append(ballot)

    if needs.rare_style_precinct_pairs:
        for pair in needs.rare_style_precinct_pairs:
            for ballot in db.ballots_by_style_precinct.get(pair, []):
                cvr_num = ballot[0].strip()
                if cvr_num not in initial_rare_cvr_nums:
                    initial_rare_cvr_nums.add(cvr_num)
                    rare_ballots.append(ballot)

    # Determine the rare contests: any contest appearing on any rare ballot.
    rare_contests: Set[str] = set()
    for ballot in rare_ballots:
        for contest_name, col_indices in db.contest_to_columns.items():
            if any(ballot[col_idx].strip() != "" for col_idx in col_indices):
                rare_contests.add(contest_name)

    # Build the common pool from non-rare styles, excluding any ballots already
    # captured by the rare-precinct set.
    rare_style_set = set(needs.rare_styles.keys())
    common_style_to_ballots_dict: Dict[str, List[List[str]]] = {}
    for style, ballots in db.ballots_by_style.items():
        if style in rare_style_set:
            continue
        # If ballot b is in initial_rare_cvr_nums, it can't be counted as an available
        # ballot in this style.
        if initial_rare_cvr_nums:
            available = [
                b for b in ballots if b[0].strip() not in initial_rare_cvr_nums
            ]
        else:
            available = list(ballots)
        if len(available) >= min_ballots:
            common_style_to_ballots_dict[style] = available

    pool = CommonPool(common_style_to_ballots_dict, min_ballots)

    # Build aggregate (Rules a, b, d).
    aggregate = build_aggregate(rare_ballots, rare_contests, pool, db, min_ballots)
    borrowed_after_rules_ab = aggregate.total_count() - len(initial_rare_cvr_nums)
    print(f"\n  Ballots borrowed for minimum counts: {borrowed_after_rules_ab}")

    # Add contrasting ballots if needed to prevent near-unanimity.
    print("\n*** Balancing near-unanimous contests.\n")
    print("  Make sure that the following constraint is met:")
    print("  - No contest in the aggregate may be near-unanimous. 'Near-unanimous'")
    print(f"    means all but {NEAR_UNANIMOUS_THRESHOLD} votes go to a single choice.")
    print()
    balance_unanimity(aggregate, pool, db)
    borrowed_after_rule_c = aggregate.total_count() - len(initial_rare_cvr_nums)
    print(f"  Ballots borrowed after unanimity balancing: {borrowed_after_rule_c}")

    # Record which CvrNumbers ended up in the aggregate (rare + borrowed).
    aggregated_cvr_nums: Set[str] = {
        ballot[0].strip() for ballot in aggregate.ballots if ballot[0].strip()
    }

    # Build the aggregate row and verify tally.
    print(
        "\n*** Verifying that vote tallies match between original and redacted files.\n"
    )
    aggregate_row = _build_aggregate_row(aggregate.ballots, db, "AGGREGATED")
    _verify_redaction_tally(db, aggregated_cvr_nums, aggregate_row)

    # Write the output file.
    # Ballots in the aggregate have vote columns replaced with '*'.
    # All ballots have geographic fields blanked per the redact_on_precinct setting.
    # The aggregate row is appended at the end in its own row.
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator=db.lineterminator)
        writer.writerow(db.version)
        writer.writerow(db.contests)
        writer.writerow(db.choices)
        writer.writerow(db.headers)
        for ballot in db.ballots:
            cvr_num = ballot[0].strip()
            if cvr_num in aggregated_cvr_nums:
                writer.writerow(_redact_ballot_row(ballot, db))
            else:
                writer.writerow(
                    _blank_geographic_fields(ballot, db, redact_on_precinct)
                )
        writer.writerow(aggregate_row)

    initial_count = len(initial_rare_cvr_nums)
    borrowed_count = len(aggregated_cvr_nums) - initial_count
    print("\n*** Redaction complete.")
    print(f"  Ballots from rare styles/precincts: {initial_count}")
    if borrowed_count > 0:
        print(f"  Ballots borrowed from common styles: {borrowed_count}")
    print(f"  Total ballots in aggregate: {len(aggregated_cvr_nums)}")
    print(f"  Output written to: {output_file}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Anonymize Cast Vote Records per Colorado C.R.S. 24-72-205.5. "
            "Ballot styles with fewer than --min-ballots ballots are aggregated "
            "to protect voter privacy."
        )
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CVR file (CSV or Parquet format).",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help=(
            "Path for the redacted output CVR file. "
            "Required in redact mode; not needed in --check mode."
        ),
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Check mode: report whether redaction is needed without writing output.",
    )
    parser.add_argument(
        "--redact-on-precinct",
        action="store_true",
        help=(
            "Treat precincts with fewer than --min-ballots ballots as rare "
            "and aggregate them, instead of simply blanking the PrecinctPortion column."
        ),
    )
    parser.add_argument(
        "--min-ballots",
        type=int,
        default=MIN_BALLOTS_DEFAULT,
        metavar="N",
        help=f"Minimum ballots required per style or precinct (default: {MIN_BALLOTS_DEFAULT}).",
    )
    parser.add_argument(
        "--headerlen",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of header columns before vote data begins. "
            "Auto-detected from the contests row if not specified."
        ),
    )
    parser.add_argument(
        "--stylecol",
        type=int,
        default=None,
        metavar="N",
        help="Column index (0-based) of the named_style field, if present.",
    )
    args = parser.parse_args()
    if args.check and args.output_file is not None:
        parser.error("output_file cannot be specified in --check mode.")
    if not args.check and args.output_file is None:
        parser.error("output_file is required when not in --check mode.")
    return args


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load the CVR file.
    try:
        db = CvrDatabase(
            args.input_file,
            args.headerlen,
            args.stylecol,
            args.redact_on_precinct,
        )
    except (ValueError, OSError) as e:
        print(f"Error reading CVR file: {e}", file=sys.stderr)
        sys.exit(1)

    # Warn if --redact-on-precinct was requested but no PrecinctPortion column exists.
    if args.redact_on_precinct and db.precinct_portion_idx is None:
        print(
            "Warning: --redact-on-precinct was requested but the CVR has no "
            "PrecinctPortion column.  The option will have no effect.",
            file=sys.stderr,
        )

    print("*** Determining if redaction is needed.")
    print()
    if args.redact_on_precinct:
        print(
            "*** Looking for rare ballot styles and rare precinct/style combinations."
        )
    else:
        print("*** Looking for rare ballot styles.")
    print()

    # Determine what redaction is needed.
    needs = check_redaction_needs(db, args.min_ballots)

    # Report leakage warnings (always, in both check and redact mode).
    for warning in needs.leakage_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    # Print rare styles and rare (precinct, style) pairs.
    if needs.rare_styles:
        total_styles = len(db.ballots_by_style)
        print(
            f"  Rare ballot styles ({len(needs.rare_styles)} of {total_styles} total):"
        )
        # Sort by ballot count, fewest first.
        for style, count in sorted(needs.rare_styles.items(), key=lambda item: item[1]):
            description = f"{count} ballot(s), {style.count('1')} contest(s)"
            ballot_types = db.ballot_types_by_style.get(style, set())
            if ballot_types:
                types_str = ", ".join(f'"{t}"' for t in sorted(ballot_types))
                description += f"  [ballot type: {types_str}]"
            print(f"    {description}")

    if needs.rare_style_precinct_pairs:
        print(
            f"\n  Rare (precinct, style) pairs "
            f"({len(needs.rare_style_precinct_pairs)}): CvrNumber(s)"
        )
        for (style, precinct), count in sorted(
            needs.rare_style_precinct_pairs.items(), key=lambda item: item[1]
        ):
            ballot_types = db.ballot_types_by_style.get(style, set())
            if ballot_types:
                style_desc = ", ".join(f'"{t}"' for t in sorted(ballot_types))
                style_desc = f"style: {style_desc}"
            else:
                style_desc = style
            pair_ballots = db.ballots_by_style_precinct.get((style, precinct), [])
            cvr_list = ", ".join(b[0].strip() for b in pair_ballots)
            print(f'    precinct "{precinct}", {style_desc}: {cvr_list}')

    # Print conclusion.
    if needs.needs_redaction():
        print("\nRedaction is needed.")
    else:
        print("No redaction needed.")

    if args.check:
        return

    print("\n*** Beginning redaction.\n")
    # Redaction mode.
    try:
        perform_redaction(
            db, needs, args.min_ballots, args.output_file, args.redact_on_precinct
        )
    except (ValueError, OSError) as e:
        print(f"Error during redaction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
