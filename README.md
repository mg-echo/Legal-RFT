# ⚖️ Legal-RFT：Enhancing Legal LLMs' Reasoning Capabilities through Reinforcement Fine-tuning

本项目为“大学生创新训练计划”项目的官方代码仓库。针对法律裁判预测（Legal Judgment Prediction, LJP）任务中存在的定性定量任务耦合、长文本检索注意力分散以及量刑逻辑一致性不足等挑战，本项目提出了一种**低资源多模型协同推理框架**。

该框架整合了**自动化数据合成**、**定性-定量级联检索增强（RAG）架构**，并创新性地针对分类（法条预测）和回归（量刑预测）任务分别引入了 **DPO偏好对齐** 与 **DAPO强化微调**，显著提升了法律大模型在复杂案情下的推理准确性与规范性。

---

## 📂 仓库结构

```text
Legal-RFT/
│
├── README.md                              # 项目总体说明文档
├── requirements_trl.txt                   # TRL及其依赖环境配置
├── requirements_llama-factory.txt         # LLaMA-Factory及其依赖环境配置
├── data/                  # 知识库 与 数据集示例
├── data_pipeline/         # 自动化数据合成 Prompt 模板
├── training/              # 模型训练代码
│   ├── sft_stage/         # 法条与量刑 SFT (基于 TRL)
│   ├── dpo_stage/         # 法条 DPO (基于 LLaMA-Factory)
│   └── dapo_stage/          # 量刑 DAPO (基于 TRL + vLLM, 包含自定义奖励函数)
└── evaluation/            # 端到端级联推理与评估脚本
```

---

## 🛠️ 环境配置
```bash
conda create -n legalrft python=3.12
conda activate legalrft
# 据训练阶段的不同，安装相应的依赖
# TRL 依赖 (SFT与DAPO)
pip install -r requirements_trl.txt
#  LLaMA-Factory 依赖 (DPO)
pip install -r requirements_llama-factory.txt
```
---

## 🚀 快速启动

### 1. 数据合成
参考 `data_pipeline/` 目录下的 Markdown 提示词文件，使用教师模型（如 DeepSeek）进行数据合成。合成后的数据按照 `data/example_dataset/` 中的 JSONL 格式存放。

### 2. 模型训练
硬件要求声明： 本项目基于中科大瀚海22超级计算系统 SLURM 调度系统实现，实验中使用了 8卡 NVIDIA A100-SXM4-80GB GPU。
依次执行以下脚本，执行模型训练：
```bash
cd training

# 冷启动 SFT
sbatch sft_stage/sft_article.sh
sbatch sft_stage/sft_penalty.sh
# 法条边界 DPO
sbatch dpo_stage/dpo_article.sh
# 刑期 DAPO
sbatch dapo_stage/dapo_penalty.sh
```

### 3. 级联评估
进入 `evaluation/` 目录，进行性能评估：
```bash
cd evaluation
python fact_law_mapping.py   # RAG 检索 Top-5 候选法条
python eval_article.py       # 定性任务评估 (法条与罪名)
python eval_penalty_tool.py  # 定量任务评估 (刑期推演)
```
---
