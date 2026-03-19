# Proposal: TRACE-Hallu (Revised v2)

## Thesis
Hallucination mitigation should move from post-hoc filtering to online intervention during generation.

## Falsifiable Questions
1. Does onset-aware intervention reduce claim-level hallucination at matched latency vs post-hoc baselines?
2. Is onset-triggered intervention better than fixed/random triggers?
3. Does the gain persist under domain shift?

## Quantitative Success Criteria
- Primary: claim-F1 `>= +3.0` absolute over strongest post-hoc baseline at matched cost.
- Secondary: no more than 10% latency increase at matched quality.

## Method
1. Convert generation trajectory into claim graph.
2. Predict hallucination onset from prefix-only features.
3. Apply intervention policy over `{retrieve, verify, revise, restart, continue}`.
4. Optimize quality under token/latency penalty.

## What Was Unreasonable Before and Is Corrected
- Causal claim without causal protocol -> now stage-gated onset/intervention design.
- Label quality unspecified -> now requires adjudicated claim set and reliability threshold.
- Cost ignored -> now explicit matched-cost comparison requirement.

## Current Gap
- Pilot implementation exists (`run_trace_hallu_pilot.py`) with first offline results.
- Full claim-level online intervention benchmark is still pending.
