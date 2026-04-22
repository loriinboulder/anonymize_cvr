# To-Do

## Code cleanup

1. **Consolidate "rules" references**
   The code mentions Rules a, b, c, d in scattered docstrings and comments with no
   single authoritative definition.  Collect them in one place (e.g., a module-level
   docstring block or a comment section above the redaction functions) and replace
   scattered mentions with references to that block.

## Design questions


2. **Re-check rare precincts after ballot borrowing**
   When ballots are borrowed from common styles for the aggregate, those styles lose
   ballots.  A precinct that was not rare before borrowing could become rare afterward
   (if the borrowed ballots happened to be the last ones for that precinct).  Decide
   whether a second pass of precinct-rarity checking is needed after aggregation, and
   if so, how to handle newly-rare precincts without triggering unbounded iteration.
