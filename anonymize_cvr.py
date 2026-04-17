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
from typing import Dict, List, Optional, Set

from cvr_utils import TempCVRFile

MIN_BALLOTS_DEFAULT = 10


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

    A ballot's style is a string of '1' and '0' characters, one per contest
    in contest_names order.  '1' means the contest is present on the ballot
    (at least one choice column is non-empty), '0' means it is absent.

    After construction the following are available:

      ballots               — all ballot rows as lists of strings
      contest_names         — ordered list of unique contest names
      ballots_by_style      — ballots grouped by style string
      ballots_by_named_style — ballots grouped by named_style value
                               (only populated when named_style_col is set)
      ballots_by_ballot_type — ballots grouped by BallotType value
                               (only populated when the column exists)
      ballots_by_precinct   — ballots grouped by PrecinctPortion value
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
            redact_on_precinct: If True, populate ballots_by_precinct for
                                per-precinct rare-style detection.
        """
        self.input_file = input_file
        self.headerlen = headerlen  # may be updated to auto-detected value in _read_file()
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

        # Ballots grouped by style string.
        self.ballots_by_style: Dict[str, List[List[str]]] = {}

        # Ballots grouped by named_style value (only when named_style_col is set).
        self.ballots_by_named_style: Dict[str, List[List[str]]] = {}

        # Ballots grouped by BallotType value (only when ballot_type_idx is set).
        self.ballots_by_ballot_type: Dict[str, List[List[str]]] = {}

        # Ballots grouped by PrecinctPortion value.
        # Only populated when redact_on_precinct is True and precinct_portion_idx is set.
        self.ballots_by_precinct: Dict[str, List[List[str]]] = {}

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

    def _validate_ballot_contents(self) -> None:
        """
        Verify that for each contest on each ballot, choice columns are either
        all empty (contest absent) or all non-empty (contest present).
        Mixed state indicates a malformed ballot row.
        """
        for i, ballot in enumerate(self.ballots):
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
        by_precinct: Dict[str, List[List[str]]] = defaultdict(list)
        named_styles_by_style: Dict[str, Set[str]] = defaultdict(set)
        ballot_types_by_style: Dict[str, Set[str]] = defaultdict(set)

        for ballot in self.ballots:
            style = self._style_for_ballot(ballot)
            by_style[style].append(ballot)

            if self.named_style_col is not None:
                named_style = ballot[self.named_style_col].strip()
                by_named_style[named_style].append(ballot)
                named_styles_by_style[style].add(named_style)

            if self.ballot_type_idx is not None:
                ballot_type = ballot[self.ballot_type_idx].strip()
                by_ballot_type[ballot_type].append(ballot)
                ballot_types_by_style[style].add(ballot_type)

            if self.redact_on_precinct and self.precinct_portion_idx is not None:
                precinct = ballot[self.precinct_portion_idx].strip()
                by_precinct[precinct].append(ballot)

        self.ballots_by_style = dict(by_style)
        self.ballots_by_named_style = dict(by_named_style)
        self.ballots_by_ballot_type = dict(by_ballot_type)
        self.ballots_by_precinct = dict(by_precinct)
        self.named_styles_by_style = dict(named_styles_by_style)
        self.ballot_types_by_style = dict(ballot_types_by_style)


# ---------------------------------------------------------------------------
# RedactionNeeds
# ---------------------------------------------------------------------------


class RedactionNeeds:
    """
    Describes what the CVR requires before it can be safely published.

    Populated by check_redaction_needs().  The redaction logic (to be added
    later) will read this to decide what work to do.

    Rare styles and rare precincts both require the same kind of treatment:
    ballots must be aggregated so no individual voter can be identified.
    Near-unanimity checking (Rule 7) applies to both and will be added here
    when that phase is implemented.
    """

    def __init__(self) -> None:
        # Styles with too few ballots.
        # Key: style string.  Value: ballot count.
        self.rare_styles: Dict[str, int] = {}

        # Precincts with too few ballots.
        # Only populated when --redact-on-precinct is requested.
        # Key: PrecinctPortion value.  Value: ballot count.
        self.rare_precincts: Dict[str, int] = {}

        # Human-readable leakage warnings.  Leakage is reported but not corrected.
        self.leakage_warnings: List[str] = []

    def needs_redaction(self) -> bool:
        """Return True if any redaction work is required."""
        return len(self.rare_styles) > 0 or len(self.rare_precincts) > 0


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

    # Check for rare precincts.
    # ballots_by_precinct is only populated when --redact-on-precinct was set.
    for precinct, ballots in db.ballots_by_precinct.items():
        if len(ballots) < min_ballots:
            needs.rare_precincts[precinct] = len(ballots)

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

    # Determine what redaction is needed.
    needs = check_redaction_needs(db, args.min_ballots)

    # Report leakage warnings (always, in both check and redact mode).
    for warning in needs.leakage_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.check:
        if needs.needs_redaction():
            print("Redaction needed.")
            if needs.rare_styles:
                total_styles = len(db.ballots_by_style)
                print(
                    f"\n  Rare ballot styles "
                    f"({len(needs.rare_styles)} of {total_styles} total):"
                )
                # Sort by ballot count, fewest first.
                for style, count in sorted(needs.rare_styles.items(), key=lambda item: item[1]):
                    description = f"{count} ballot(s), {style.count('1')} contest(s)"
                    # Add ballot type(s) as a human-readable identifier, if available.
                    ballot_types = db.ballot_types_by_style.get(style, set())
                    if ballot_types:
                        types_str = ", ".join(f'"{t}"' for t in sorted(ballot_types))
                        description += f"  [ballot type: {types_str}]"
                    print(f"    {description}")

            if needs.rare_precincts:
                print(f"\n  Rare precincts ({len(needs.rare_precincts)}):")
                for precinct, count in sorted(
                    needs.rare_precincts.items(), key=lambda item: item[1]
                ):
                    print(f'    "{precinct}": {count} ballot(s)')
        else:
            print("No redaction needed.")
        return

    # Redaction mode: not yet implemented.
    print("Redaction is not yet implemented.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
