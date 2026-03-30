# Round 2 Review (Codex GPT-5.4)

## Score: 6/10 (up from 3/10)
## Verdict: No longer RETHINK. Borderline-to-competitive.

## Remaining Gaps
1. **PPO training/eval mismatch**: Train with simulated rewards, eval online. Need online rollouts or justify offline-to-online transfer.
2. **Action space fairness**: Normalize for compute, tokens, external knowledge access.
3. **Label validation**: Need human-audited subset (200-500 cases).
4. **Detector metrics**: Add AUPRC, calibration/ECE, onset lead-time, false-positive burden.
5. **Single model**: Add one cross-family experiment.
6. **Completeness/helpfulness**: Don't improve just by saying less. Need abstention metric.
7. **Rename**: CHI → PHI.

## Priority Actions
1. Frame as offline-to-online transfer, show correlation between offline reward and online gains
2. Add budget-matched comparisons (tokens, compute, action frequency)
3. Add detector calibration metrics
4. Add completeness scoring
5. Rename to PHI
