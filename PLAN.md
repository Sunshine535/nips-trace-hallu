# Plan: TRACE-Hallu (Stage-Gate v2)

## Gate 0: Data and Label Integrity (Partial)
- [x] Added trace-driven pilot dataset path from `01_adathink` per-sample outputs.
- [ ] Build claim extraction and adjudication pipeline.
- Go criterion: inter-annotator kappa `>= 0.70`.

## Gate 1: Onset Detector (Pilot done, full pending)
- [x] Pilot onset-risk detector implemented (`run_trace_hallu_pilot.py`).
- [x] Pilot trigger policy evaluated against fixed budgets.
- [ ] Full onset detector on claim-level labels.
- Go criterion: onset AUPRC `>= +5%` relative over best trigger baseline.

## Gate 2: Online Policy (Pending)
- [x] Offline proxy intervention policy on GSM8K traces.
- [ ] True online controller over generation actions.
- Go criterion: claim-F1 `>= +3` absolute vs strongest baseline at matched cost.

## Gate 3: Robustness
- Evaluate closed->open domain transfer and prompt perturbations.
- Go criterion: at least 70% of Gate 2 gain retained under shift.

## Gate 4: Paper Package
- Main table, ablations, error taxonomy, reproducibility artifact.

## Kill Criteria
- If onset detector cannot beat fixed/random triggers, pivot to confidence-triggered intervention without onset claim.
