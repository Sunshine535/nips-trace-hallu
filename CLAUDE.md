# Project: nips-trace-hallu

## Project goal

PHI: Predictive Hallucination Intervention via Onset Detection and Graduated Response — 收集 hidden-state 轨迹（带 claim-level NLI 幻觉标签），训练 onset detector，PPO 干预策略，对比 DoLa/ITI/SelfCheckGPT 等 baseline。

## Key models

- `Qwen/Qwen3.5-9B` — 主实验模型
- `Llama-3-8B` — 跨模型迁移实验
- `microsoft/deberta-v3-large-mnli` — NLI claim 验证

## Key datasets

- TruthfulQA — 幻觉评测
- HaluEval — 幻觉评测
- FaithDial — 对话忠实度

## Repo map

- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排（Stage 1–6）
  - `collect_traces.py` — 收集轨迹 + hidden states + claim-level NLI 标注
  - `extract_hidden_states.py` — 从已有 JSONL 抽取 hidden states
  - `train_onset_detector.py` — 训练 onset detector（单层探针 + 多层集成）
  - `train_intervention_policy.py` — 离线 PPO 干预策略（MLP, 5D→5 actions）
  - `train_policy_online.py` — 在线 PPO 策略精炼（真实 rollout）
  - `eval_chi.py` — PHI 全面评估（含 6 种 baseline 对比）
  - `analyze_offline_online_correlation.py` — 离线/在线奖励相关性分析
  - `run_ablations.py` — 消融配置矩阵生成
  - `gpu_utils.sh` — GPU 检测 + batch size 自动缩放
- `src/` — 核心模块
  - `onset_detector.py` — 线性探针 + 多层集成检测器
  - `intervention_actions.py` — 5 类干预动作 + 执行器
  - `claim_labeler.py` — Claim 提取 + NLI 幻觉标注
  - `factuality_eval.py` — 多评判器事实性评估 + bootstrap CI
  - `baselines.py` — DoLa, ITI, SelfCheckGPT 实现
  - `detector_calibration.py` — AUPRC, ECE, lead-time 校准
  - `completeness_eval.py` — 完整性、有用性、弃权率评估
  - `budget_eval.py` — 预算匹配 Pareto 评估
  - `rule_policies.py` — 规则策略 baseline（用于消融）
- `configs/trace_config.yaml` — 实验配置
- `tests/test_trace_hallu.py` — 单元测试

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# 一键全流程
bash run.sh

# 后台运行
nohup bash run.sh > run.log 2>&1 &

# 强制重跑
FORCE_RERUN=1 bash run.sh
```

## Experiment phases

| Stage | 内容 |
|-------|------|
| 1 | 收集/抽取 hidden-state 轨迹（claim-level NLI 标注） |
| 2 | 训练 onset detector（单层探针 + 多层集成） |
| 3 | 离线 PPO 干预策略训练（MLP, 5D state → 5 actions） |
| 3b | 在线 PPO 策略精炼（真实 LLM rollout + NLI 奖励） |
| 4 | 全面评估（PHI vs DoLa, ITI, SelfCheckGPT 等） |
| 5 | 消融实验（阈值 sweep + 单层 detector + 预算分析） |
| 6 | 汇总 + bootstrap CI + 显著性检验 |

## Data and outputs

- 轨迹数据: `data/traces/`
- Onset detector: `checkpoints/detector/`
- 干预策略: `checkpoints/intervention_policy/`
- 评估结果: `results/`
- 日志: `logs/`
- 研究细化: `refine-logs/`

## Environment

- Python 3.10+, PyTorch 2.x (CUDA 12.x)
- 关键依赖: transformers, datasets, accelerate, peft, h5py, scikit-learn, evaluate, wandb, sentencepiece

## Remote server

- SSH: `ssh -p 30022 wujn@root@ssh-362.default@222.223.106.147`
- GPU: 待确认
- Code dir: `/gfs/space/private/wujn/Research/nips-trace-hallu`
- Background: `screen -dmS trace-hallu bash -c '...'`
- HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`
