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
  - `extract_hidden_states.py` — 抽取 hidden states
  - `collect_traces.py` — 收集轨迹（claim-level NLI 标注）
  - `train_onset_detector.py` — 训练 onset detector
  - `train_intervention_policy.py` — PPO 干预策略
  - `eval_chi.py` — PHI 在线评估（含基线对比）
  - `run_trace_generation.sh` — 轨迹生成封装
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
  - `onset_detector.py` — 线性探针 + 多层集成检测器
  - `intervention_actions.py` — 5 类干预动作 + 执行器
  - `claim_labeler.py` — Claim-level NLI 幻觉标注
  - `factuality_eval.py` — Claim-level 事实性评估 + 置信区间
  - `baselines.py` — DoLa, ITI, SelfCheckGPT 实现
  - `detector_calibration.py` — AUPRC, ECE, lead-time 校准
  - `completeness_eval.py` — 完整性、有用性、弃权率评估
- `configs/trace_config.yaml` — 实验配置
- `results/` — 实验输出
- `refine-logs/` — ARIS 研究细化历史

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
| 3 | PPO 干预策略训练 |
| 4 | 在线评估（PHI vs DoLa, ITI, SelfCheckGPT 等） |
| 5 | 消融实验（阈值 + 单层 detector + 预算分析） |
| 6 | 汇总 + 统计检验 |

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
