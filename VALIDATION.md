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


## Figure/decoder revision — 2026-11-04

The work is split into three commits on main. No full research simulation was
run for this revision; the user runs the new experiments locally.

- Plot/smoke revision: 12 Phase 2 tests passed on the exact staged tree.
- Decoder ablation/paired-record revision: 14 Phase 2 tests passed on the exact
  staged tree. Tiny fixtures verify list-only retries and per-frame totals.
- Final revision: 15 Phase 2 tests passed. Fig5 protocol checks Alice=Bob, exact rounded mismatch counts over
  all N key bits, and the structured complement-key case at N=128/256.
- Replotting the supplied completed run leaves metrics.csv byte-for-byte intact.
  Smoke/single-SNR runs are rejected for performance curves; zero-error bounds
  are not inserted as measured rates.
- The dense/expanded/Fig5 JSON plans are validated without running them. Fig5
  uses N128/K64 and N256/K64, 0..18 dB, CA-SCL L8; the reference paper uses SC/AWGN,
  so the new configuration is not claimed to reproduce its numerical curves.
