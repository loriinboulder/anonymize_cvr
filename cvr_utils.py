#!/usr/bin/env python3
"""
Utility functions for reading CVR files in different formats.

Supports both CSV and Parquet formats, with automatic format detection
and conversion to the expected CSV format.
"""

import csv
import tempfile
import os
from typing import Any, Dict, IO, List, Optional

try:
    import pandas as pd  # type: ignore[import]

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def is_parquet_file(file_path: str) -> bool:
    """
    Check if a path is a Parquet file or a Hive-partitioned Parquet directory.

    Args:
        file_path: Path to the file or directory

    Returns:
        True if file appears to be a Parquet file or directory of Parquet files
    """
    if file_path.lower().endswith(".parquet"):
        return True
    if os.path.isdir(file_path):
        for root, dirs, files in os.walk(file_path):
            for f in files:
                if f.lower().endswith(".parquet"):
                    return True
    return False


def _build_contest_candidates(partition_files: List[str]) -> Dict[str, List[str]]:
    """
    Pass 1: scan all partition files reading only contest/candidate/isVote columns
    to determine the full set of contests and candidates across the dataset.
    Returns a dict mapping contest name -> sorted list of candidate names.
    """
    # Use a dict of sets so we can accumulate candidates per contest without duplicates.
    contest_candidates_raw: Dict[str, set] = {}
    print(
        f"Pass 1: scanning contests and candidates across {len(partition_files)} partitions...",
        flush=True,
    )
    for i, pf in enumerate(partition_files, 1):
        if i % 50 == 0 or i == len(partition_files):
            print(f"  Scanned {i}/{len(partition_files)} partitions...", flush=True)
        part_df = pd.read_parquet(pf, columns=["contest", "candidate", "isVote"])
        votes_part = part_df[part_df["isVote"]]
        for contest, candidate in zip(votes_part["contest"], votes_part["candidate"]):
            if contest not in contest_candidates_raw:
                contest_candidates_raw[contest] = set()
            contest_candidates_raw[contest].add(candidate)
        del part_df, votes_part

    contest_candidates: Dict[str, List[str]] = {}
    for contest in sorted(contest_candidates_raw.keys()):
        contest_candidates[contest] = sorted(contest_candidates_raw[contest])
    print(f"Found {len(contest_candidates)} contests.", flush=True)
    return contest_candidates


def _build_header_rows(contest_candidates: Dict[str, List[str]]) -> tuple:
    """Build the four CSV header rows from the contest/candidate mapping."""
    contests = list(
        contest_candidates.keys()
    )  # already sorted by _build_contest_candidates

    version_row = ["Parquet CVR", "V1"] + [""] * 6

    contests_row = [""] * 8
    for contest in contests:
        contests_row.extend([contest] * len(contest_candidates[contest]))

    choices_row = [""] * 8
    for contest in contests:
        choices_row.extend(contest_candidates[contest])

    headers_row = [
        "CvrNumber",
        "TabulatorNum",
        "BatchId",
        "RecordId",
        "ImprintedId",
        "CountingGroup",
        "PrecinctPortion",
        "BallotType",
    ]
    for contest in contests:
        headers_row.extend(contest_candidates[contest])

    return version_row, contests_row, choices_row, headers_row


def _write_voter_rows(
    writer: Any,
    votes_df: "pd.DataFrame",
    contests: List[str],
    contest_candidates: Dict[str, List[str]],
    starting_idx: int,
    ballot_contest_set: Optional[set] = None,
) -> int:
    """
    Write one CSV row per voter from an already-filtered (isVote=True) DataFrame.
    Returns the updated CvrNumber counter after writing all voters.

    ballot_contest_set, if provided, is the set of all contests on the ballot for
    every voter in this batch, determined from all rows including isVote=False
    (undervotes and overvotes).  When omitted, contest presence is derived from
    the voted rows alone, which incorrectly marks undervoted contests as absent.
    """
    idx = starting_idx
    for voter_id, voter_votes in votes_df.groupby("voter_id"):
        idx += 1
        if idx % 5000 == 0:
            print(f"  Voter {idx:,}...", flush=True)

        precinct_portion = str(int(voter_votes["precinctPortionId"].iloc[0]))
        voted_pairs = set(zip(voter_votes["contest"], voter_votes["candidate"]))
        if ballot_contest_set is not None:
            voter_contest_set = ballot_contest_set
        else:
            voter_contest_set = set(voter_votes["contest"])

        row = [
            str(idx),  # CvrNumber
            "1",  # TabulatorNum
            "1",  # BatchId
            str(idx),  # RecordId
            voter_id,  # ImprintedId (use voter_id)
            "1",  # CountingGroup
            precinct_portion,  # PrecinctPortion
            "",  # BallotType
        ]

        for contest in contests:
            for candidate in contest_candidates[contest]:
                if (contest, candidate) in voted_pairs:
                    row.append("1")
                elif contest in voter_contest_set:
                    row.append("0")
                else:
                    row.append("")

        writer.writerow(row)

    return idx


def convert_parquet_to_csv_format(parquet_file: str, csv_output: str) -> None:
    """
    Convert a Parquet format CVR file to CSV format expected by this tool.

    The Parquet file is in long format (one row per candidate per contest per voter),
    while the CSV format is in wide format (one row per voter with columns for each candidate).

    For a directory of Hive-partitioned files, uses a two-pass approach to avoid
    loading the entire dataset into memory at once:
      Pass 1 — read only contest/candidate/isVote columns to build the CSV headers.
      Pass 2 — read one partition at a time, write its voters, then discard it.

    Args:
        parquet_file: Path to input Parquet file or Hive-partitioned directory
        csv_output: Path to output CSV file

    Raises:
        ImportError: If pandas is not available
        ValueError: If the Parquet file doesn't have expected columns
    """
    if not PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required to read Parquet files. "
            "Install with: pip install pandas pyarrow"
        )

    if not os.path.isdir(parquet_file):
        # Single file: load it all at once (single files are small enough).
        df = pd.read_parquet(parquet_file)
        required_cols = [
            "voter_id",
            "contest",
            "candidate",
            "isVote",
            "precinctPortionId",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Parquet file missing required columns: {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )
        ballot_contest_set = set(df["contest"].unique())
        votes_df = df[df["isVote"]].copy()
        del df
        contest_candidates = {}
        for contest in sorted(ballot_contest_set):
            candidates = sorted(
                votes_df[votes_df["contest"] == contest]["candidate"].unique()
            )
            if candidates:
                contest_candidates[contest] = candidates
        contests = list(contest_candidates.keys())
        version_row, contests_row, choices_row, headers_row = _build_header_rows(
            contest_candidates
        )
        n_voters = votes_df["voter_id"].nunique()
        print(f"Found {n_voters:,} voters and {len(contests)} contests.", flush=True)
        print(f"Writing {n_voters:,} voter rows to {csv_output}...", flush=True)
        with open(csv_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(version_row)
            writer.writerow(contests_row)
            writer.writerow(choices_row)
            writer.writerow(headers_row)
            _write_voter_rows(
                writer,
                votes_df,
                contests,
                contest_candidates,
                0,
                ballot_contest_set=ballot_contest_set,
            )
        print("Done.", flush=True)
        return

    # Directory of Hive-partitioned files: two-pass approach.
    partition_files = []
    for root, dirs, files in os.walk(parquet_file):
        for fname in files:
            if fname.lower().endswith(".parquet"):
                partition_files.append(os.path.join(root, fname))
    partition_files.sort()
    print(f"Found {len(partition_files)} parquet partition files.", flush=True)

    # Verify required columns exist using the first partition.
    sample_df = pd.read_parquet(partition_files[0])
    required_cols = ["voter_id", "contest", "candidate", "isVote", "precinctPortionId"]
    missing_cols = [col for col in required_cols if col not in sample_df.columns]
    if missing_cols:
        raise ValueError(
            f"Parquet file missing required columns: {missing_cols}. "
            f"Found columns: {list(sample_df.columns)}"
        )
    del sample_df

    # Pass 1: scan all partitions for contest/candidate info to build headers.
    contest_candidates = _build_contest_candidates(partition_files)
    contests = list(contest_candidates.keys())
    version_row, contests_row, choices_row, headers_row = _build_header_rows(
        contest_candidates
    )

    # Pass 2: read one partition at a time, writing each voter's row to the CSV.
    print(f"Pass 2: writing voter rows to {csv_output}...", flush=True)
    idx = 0
    with open(csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(version_row)
        writer.writerow(contests_row)
        writer.writerow(choices_row)
        writer.writerow(headers_row)

        for i, pf in enumerate(partition_files, 1):
            print(f"  Partition {i}/{len(partition_files)}: {pf}", flush=True)
            part_df = pd.read_parquet(pf)
            partition_contest_set = set(part_df["contest"].unique())
            votes_part = part_df[part_df["isVote"]].copy()
            del part_df
            idx = _write_voter_rows(
                writer,
                votes_part,
                contests,
                contest_candidates,
                idx,
                ballot_contest_set=partition_contest_set,
            )
            del votes_part

    print(f"Done. Wrote {idx:,} voter rows.", flush=True)


class TempCVRFile:
    """
    Context manager for handling CVR files that may need conversion from Parquet to CSV.

    If the input file is a Parquet file, it converts it to a temporary CSV file.
    The temporary file is automatically cleaned up when the context exits.

    Usage:
        with TempCVRFile(input_path) as csv_path:
            # Use csv_path to read the CVR data
            process_cvr(csv_path)
    """

    def __init__(self, input_path: str):
        """
        Initialize the context manager.

        Args:
            input_path: Path to input CVR file (CSV or Parquet)
        """
        self.input_path = input_path
        self.temp_file: Optional[IO[str]] = None
        self.temp_path: Optional[str] = None
        self.is_parquet = is_parquet_file(input_path)

    def __enter__(self) -> str:
        """
        Enter the context, converting Parquet to CSV if needed.

        Returns:
            Path to CSV file (either original or temporary converted file)
        """
        if self.is_parquet:
            # Create temporary CSV file
            self.temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            )
            self.temp_path = self.temp_file.name
            self.temp_file.close()

            # Convert Parquet to CSV
            convert_parquet_to_csv_format(self.input_path, self.temp_path)
            return self.temp_path
        else:
            # Already CSV, return original path
            return self.input_path

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """
        Exit the context, cleaning up temporary file if created.
        """
        if self.temp_path is not None:
            try:
                os.unlink(self.temp_path)
            except Exception:
                pass  # Ignore cleanup errors
