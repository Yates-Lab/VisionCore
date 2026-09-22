# Figure 4 passband audit results

These compact reports are exact copies of the completed analyses in
`outputs/passband_comparison_20260914/`. The adjacent
`../passband_comparison.json` binds their hashes to the selected checkpoint
and supplies the manuscript statistics.

- [Minimal audit findings](normalized_overlap/FINDINGS.md): the movement,
  power, and normalized-overlap comparisons used in the revised text.
- [Full comparison findings](FINDINGS.md): additional controls and limits.
- `summary.json` and `audit.json`: original comparison results and checks.
- `normalized_overlap/summary.json`: the normalized-overlap results and
  checks; its `design.json` records the follow-up design.

The large source caches, per-unit prediction archives, and rendered PDFs
remain in the local output directories. Reproduction and full source
verification require those caches, as do the existing production-figure
statistics. Commands and the statistical design are documented in
`jake/passband_comparison/README.md` and `MINIMAL_AUDIT.md`.
