# Round 1 Review (Codex GPT-5.4)

## Score: 3/10
## Verdict: RETHINK

## Key Issues (Prioritized)
1. **Labels invalid**: Word overlap ≠ hallucination onset. Need claim-level verified annotations.
2. **Evaluation invalid**: String matching ≠ factuality. Need claim extraction + verification.
3. **PPO not real**: Simulated rewards over offline traces. Need online intervention evaluation.
4. **"Causal" overclaimed**: Correlational probing, not causal. Remove or earn with intervention evidence.
5. **Below conference bar**: No SOTA baselines, no multi-seed, no significance tests.

## Minimum Fixes for Score >= 9
1. Replace heuristic labels with claim-level span-level verified annotations
2. Replace string-overlap factuality with proper protocol (claim extraction + verification, LLM-as-judge)
3. Evaluate interventions online in live decoding loop
4. Benchmark against strong baselines (DoLa, ITI, SelfCheckGPT, Self-RAG, verifier/reranker)
5. Remove "causal" or earn it with real mechanistic intervention evidence
6. Show robustness across models, domains, seeds
7. Formalize control problem (optimal stopping, POMDP)

## Best Paper Suggestions
- Make paper about benchmark first, method second
- Prove early warning value with calibrated uncertainty
- Real cost-quality tradeoffs online
- Cross-model generalization
- Theory story (hazard model, POMDP, optimal stopping)
