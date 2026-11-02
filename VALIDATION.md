# Technical validation — 2026-11-02

Executed on Linux, Python 3.12, CPU. No full Phase 1 or Phase 2 research experiment
was run for this revision. Windows execution remains to be checked by the user.

- Phase 1: 4 unittest cases passed (frequency offsets, padding/delay metrics,
  empty PDP, no-path failure before CSI export).
- Phase 2: 11 unittest cases passed. Includes all 12 expanded N,K configurations
  as noiseless roundtrip subcases; reference CA-SCL comparison with CRC; retry
  telemetry; wrong-key scope/count; frozen-only inverse-transform control;
  full-band mapping; analytical BPSK/gain checks; zero-error statistics;
  configuration validation; end-to-end fixture output with an incomplete batch.
- The end-to-end fixture used N=32, K=16, CRC6 and only 3 frames on artificial
  1x1 CSI. The run was labeled `technical-smoke` and temporary outputs were removed
  by the test harness. It is not evidence for research performance.
- Both review and expanded config plans validated without running experiments.
- Python compile checks, shell syntax check and `git diff --check` passed.

Commands to reproduce are in RUN_VSCODE.md. Tests generate no committed research
figures or datasets. CSV fields preserve measured zero errors and provide a
separately labeled upper bound; there are no prewritten performance conclusions.
