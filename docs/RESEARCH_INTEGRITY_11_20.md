# Research family and label APIs (11, 12, 14, 17–19)

Install `.[research]` for the optional shared quant-lab statistical/nested APIs.

* `combinatorial.family_statistics`: shared DSR and original CSCV/PBO; pass complete excess
  return matrices, explicit periods_per_year and block count. It is not chronological OOS.
* `purged_combinatorial_splits(starts, ends, blocks=6, test_blocks=2, embargo=...)`: tests all
  combinations, purges exact label-interval overlaps and adds post-label-end embargo. It keeps
  train/test/purged/embargoed indices and identifies a different estimand from original CSCV.
  Disjoint test blocks do not incorrectly purge every row between them.
* `dependent_mean_evidence`: common-index stationary/circular bootstrap sensitivity plus HAC.
  Do not treat correlated stock-day observations as independent samples.
* `nested_factor_selection`: fit learned direction/window/neutralization/model settings inside
  the inner training callback and refit the selected recipe on outer training. Callbacks receive
  bounded copies, but must not use global future data. Returned scores are not funded accounts.

`research.factor_report(..., terminal_labels=None, sessions=None)` now returns `label_samples`
and `label_coverage` for each horizon. Missing/delisted/immature selected rows remain visible.
Terminal evidence is per-sample holding-period return with source, availability and optionally
horizon; sample IDs are `symbol:YYYY-MM-DD`. Use the same economic return-index/share basis.
If available_at is absent, the report explicitly uses a descriptive end-of-day availability
assumption; for strict PIT research supply actual availability and an exchange calendar.
Daily signal decisions use the timezone-naive session date at 23:59 UTC. When actual
`available_at` is supplied, every rolling dependency must be visible by that decision time;
late rows do not backfill earlier signals, and missing sessions do not compress windows.
The default calendar is the union of supplied dates, not proof of complete exchange coverage.

Existing observed-label IC remains descriptive. The new denominator and status breakdown
expose which securities were excluded from its numeric calculation. Unknown terminal cash
cannot be silently replaced with zero. See QDK `docs/SELECTION_LABELS.md` for the shared contract.

Regression tests preserve future-prefix invariance of v2 turnover/Amihud factors and show
that an evidenced terminal −100% sample survives factor reporting. No statistical result
automatically promotes a factor or authorizes a trade.
