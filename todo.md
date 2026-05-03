# To-Do

## Code cleanup

1. **Consolidate "rules" references**
   The code mentions Rules a, b, c, d in scattered docstrings and comments with no
   single authoritative definition.  Collect them in one place (e.g., a module-level
   docstring block or a comment section above the redaction functions) and replace
   scattered mentions with references to that block.

## Design questions


3. **Handle "blocked style" case during ballot borrowing**

   Currently, when `best_candidate_for()` returns None, the code warns and breaks.
   This conflates two distinct situations:

   - **Florence case**: the contest appears on zero common-style ballots — all its
     ballots are already in the aggregate.  Warning is correct; nothing more to do.
   - **Blocked-style case**: the contest appears on common styles, but every such
     style is at exactly `min_ballots` ballots — one borrow would make it rare, so
     `_best_for_contests` skips it.  A solution exists: pull the whole style in.

   **Why blocked styles are small**: After the borrow loop, any style remaining in
   `_styles` with `len(rows) <= min_ballots` has exactly `min_ballots` ballots (the
   pool invariant).  Pulling it adds exactly `min_ballots` ballots to the aggregate
   — small and bounded.  (The original style could have been as large as
   `2 * min_ballots - 2` before borrowing began, but by the time we pull it only
   `min_ballots` remain.)

   **Proposed fix**: in `build_aggregate()`, when `best_candidate_for()` returns None,
   before warning, call a new `CommonPool.pull_blocked_style_for(needed_contests, db)`
   method:
   - Search `_styles` for styles with `len(rows) <= min_ballots` that cover at least
     one still-needed contest.
   - Pick the one with the best coverage (most needed contests covered).
   - Remove it from `_styles` and return all its ballots.
   - Caller adds all ballots to the aggregate and continues the loop.
   - If nothing is found, fall through to the Florence warning.

   **No cascade concern**: the pulled style has exactly `min_ballots` ballots; once
   all are in the aggregate every one of its contests is automatically covered.

   Sketch of new `CommonPool` method:
   ```python
   def pull_blocked_style_for(self, needed_contests, db):
       best_sig, best_coverage = None, 0
       for style_sig, rows in self._styles.items():
           if len(rows) > self.min_ballots:
               continue
           covered = sum(
               1 for c in needed_contests
               if any(_ballot_has_contest(r, c, db.contest_to_columns) for r in rows)
           )
           if covered > best_coverage:
               best_coverage, best_sig = covered, style_sig
       if best_sig is None:
           return None
       return (best_sig, self._styles.pop(best_sig))
   ```

2. **Re-check rare precincts after ballot borrowing**
   When ballots are borrowed from common styles for the aggregate, those styles lose
   ballots.  A precinct that was not rare before borrowing could become rare afterward
   (if the borrowed ballots happened to be the last ones for that precinct).  Decide
   whether a second pass of precinct-rarity checking is needed after aggregation, and
   if so, how to handle newly-rare precincts without triggering unbounded iteration.
