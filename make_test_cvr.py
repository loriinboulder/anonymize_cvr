"""
make_test_cvr.py — Generate test CVR files for anonymize_cvr.py

Starting from tc_ElPaso2023_wstyle.csv, produces tc_ElPaso2023_unredacted.csv
with the following anonymization challenges deliberately built in:

  1. Two rare styles (< min_ballots ballots each):
       Style 3  — 4 ballots: Florence RE-2 + Prop HH + Prop II
       Style 22 — 8 ballots: Academy SD 20 + CS Ballot 2A + Prop HH + Prop II

  2. Per-contest minimums (step 3):
       Academy SD 20 and CS 2A contests appear only on style 22 (8 ballots < 10)
       → code must borrow from common styles to reach 10 per contest.
       Florence RE-2 appears only on style 3 and on no common style
       → unsolvable case; code should warn and continue.

  3. Unanimous vote triggering step 5:
       All style 3 and style 22 ballots vote No/Against on Prop HH.
       The common styles that will be borrowed for step 3 (styles 10-13, 19, 21)
       are also set to all-No on Prop HH, so the aggregate remains unanimous
       after step 3 borrowing.  Step 5 must then find a Yes/For ballot from
       one of the untouched common styles (e.g. style 1 or 6).

Usage:
    python make_test_cvr.py [input_csv [output_csv]]

Defaults:
    input  = tc_ElPaso2023_wstyle.csv
    output = tc_ElPaso2023_unredacted.csv
"""

import csv
import sys


def make_test_cvr(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        version = next(reader)
        contests = next(reader)
        choices = next(reader)
        headers = next(reader)
        rows = [r for r in reader if r]

    # Prop HH column indices (0-based):
    #   col 98 = Yes/For,  col 99 = No/Against
    prop_hh_yes = 98
    prop_hh_no = 99

    # Force all ballots in these styles to vote No/Against on Prop HH.
    #
    # Rare styles — must be unanimous so step 5 triggers:
    #   style 22 (style 3 already votes all-No in the source file)
    #
    # Common styles that will be borrowed from in step 3 — must also be
    # all-No so the aggregate stays unanimous after borrowing:
    #   styles 10-13 (have City of Colorado Springs Ballot Issue 2A)
    #   styles 19, 21 (have Academy SD 20 contests)
    force_no_styles = {"22", "10", "11", "12", "13", "19", "21"}

    changed = 0
    for row in rows:
        if row[6] in force_no_styles and row[prop_hh_yes] == "1":
            row[prop_hh_yes] = "0"
            row[prop_hh_no] = "1"
            changed += 1

    print(f"Changed {changed} ballots to No/Against on Prop HH.")

    # Verify key styles
    for style, expected_yes, expected_no_min in [
        ("3", 0, 4),
        ("22", 0, 8),
        ("1", 1, 1),   # untouched — must still have Yes voters for step 5
        ("6", 1, 1),
    ]:
        style_rows = [r for r in rows if r[6] == style]
        yes = sum(1 for r in style_rows if r[prop_hh_yes] == "1")
        no = sum(1 for r in style_rows if r[prop_hh_no] == "1")
        if yes < expected_yes or no < expected_no_min:
            raise ValueError(
                f"Style {style}: unexpected Prop HH distribution "
                f"Yes={yes} No={no} (expected Yes>={expected_yes}, No>={expected_no_min})"
            )
        print(f"  Style {style:3s} ({len(style_rows):3d} ballots): Prop HH Yes={yes} No={no}")

    # Renumber CvrNumbers sequentially starting from 1
    for seq, row in enumerate(rows, start=1):
        row[0] = str(seq)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(version)
        writer.writerow(contests)
        writer.writerow(choices)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "tc_ElPaso2023_wstyle.csv"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "tc_ElPaso2023_unredacted.csv"
    make_test_cvr(input_csv, output_csv)
