# Changelog

## 1.1.0 — extended version for Lecture Notes in Networks and Systems

Defaults are unchanged: `run.py --scenario_a --scenario_c` reproduce the version 1.0 outputs bit-for-bit,
and `run.py --scenario_b --theta_b 0.70` reproduces the version 1.0 Scenario B run.

### Specification/code conformance fixes
- **D1** The per-exposure load increment (`load_from_exposure`, λ_L = 0.05) was implemented but absent from the
  written specification; it is now Eq. (12) of the extended paper.
- **D2** The sharing threshold was a hard-coded literal `0.8`; it is now `Params.tau_share` (Table 1: τ_share).
- **D3** `run_scenario_b()` silently overrode θ_C = θ_O = 0.70 although the conference figure was generated at the
  Table 1 value 0.50. The threshold is now an explicit `theta` argument (default: config value) and a CLI flag
  `--theta_b`.

### New modules
- `audience_dt/scenarios.py` — Scenarios A/B/C as importable functions + scalar diagnostics per scenario.
- `audience_dt/invariants.py` — executable invariants I1–I8 shared by the tests and the results script.
- `audience_dt/ablation.py` — component ablations (`route_mode`, `strength_mode`, `gate_mode="none"`, γ₁ = 0).
- `audience_dt/sensitivity.py` — Morris elementary-effects screening (self-contained).
- `audience_dt/calibration.py` — synthetic parameter-recovery experiment (identifiability check).
- `experiments/run_experiments.py`, `experiments/make_figures.py` — full reproduction pipeline.
- `tests/` — 47 pytest tests (invariants, unit properties, reproducibility, qualitative claims).

### Simulator
- `simulate(..., trace=list)` optional per-agent tracing hook (off by default).
- New `Params` fields with full-model defaults: `tau_share`, `route_mode`, `pc_fixed`, `strength_mode`;
  `gate_mode` accepts `"none"`.

## 1.0.0 — SIMULTECH 2026 release
Initial release.
