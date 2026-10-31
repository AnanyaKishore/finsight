# Test notes

Hand-written context for results in the generated test report. `scripts/test_report.py`
copies this file into section 5 of `TEST_REPORT.md`; the measured values above are never edited.

### AC-11 — scheduled pipeline runs

The only failed scheduled run is `5ec75aa8d7f1`, the scheduler's first run, on
27 Oct 2026 at 13:21 UTC (Day 7, while the scheduler was being set up).

- **What failed:** the `data` step's unit-test gate (`finsight.ci.run_unit_tests`).
  One test, `test_runs.py::test_stage_run_carries_the_scheduler_run_id`, failed:
  the scheduler sets `FINSIGHT_PIPELINE_RUN_ID` for every command it runs, and the
  test read that variable instead of assuming it was unset. This was a test-isolation
  bug, not a pipeline fault. The gate worked as designed and stopped the run before
  any data changed; later steps were skipped.
- **Fix:** the same afternoon, in the Day 7 test-isolation fix: the test now clears
  the variable before checking it.
- **Since then:** every scheduled run has succeeded (4 of 4 at the time of writing).

The 90 % target is unchanged. With so few runs one failure weighs 20 %, so the
criterion passes once at least 9 of 10 scheduled runs in the 30-day window succeed
(five more successful daily runs), or when this run leaves the window on 26 Nov.
