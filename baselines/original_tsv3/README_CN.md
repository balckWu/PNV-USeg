# TaskSegment 论文实验工程（整洁重构版）

本工程以已经跑通的 TaskSegment、消融和官方对比代码为基础，只统一实验配置、调度、日志和结果管理，不改变模型核心实现。

## 1. 正式实验范围

默认配置：`configs/paper_3seeds.json`

- 随机种子：`42, 3407, 2026`
- 物理 GPU：`0, 1, 2, 4, 5, 6`
- 实际训练 batch size：`4`
- 梯度累积：`4`
- 有效 batch size：`16`
- 每轮 domain-balanced iteration：`1500`
- 最大训练轮数：`150`
- warm-up：`5`
- early stopping patience：`20`
- 固定阈值：`0.5`
- 测试后处理：关闭
- 统一评价网格：`512×512`

共 11 个设置、33 次独立训练：

1. TaskSegment Full
2. TaskSegment w/o semantic text
3. TaskSegment w/o visual prototype
4. TaskSegment w/o dense guidance
5. U-Net
6. U-Net++
7. Swin-Unet（官方代码适配）
8. TransUNet（R50-ViT-B/16，ImageNet-21k 预训练）
9. CLIPSeg（decoder-only fine-tuning）
10. LViT（native 224；domain-level prompts；512 评价）
11. SAM-Med2D（prompt-free adapted 512）

三种子不是 MIA 的强制条款，但对于正式深度学习分割比较，三个独立种子是合理的最低配置。主表报告 mean ± SD，并保存逐病例结果用于配对统计。

## 2. 目录

```text
configs/                 单一论文配置
paperlab/                调度、数据库、结果导出和统计
scripts/paper.py         唯一实验管理入口
scripts/train*.py        模型训练入口
scripts/predict*.py      模型预测入口
tasksegment/             TaskSegment 与共享数据/指标实现
third_party/             固定版本官方对比源码
paper_artifacts/         正式运行输出（自动创建）
```

旧式的一次性 `run_gpu*.sh`、补丁脚本和重复汇总脚本不再作为正式入口。

## 3. 服务器部署

建议在旧工程旁新建目录，不覆盖已完成实验：

```bash
cd /tank/wk
unzip TSV2_paper_clean_v4.zip
cd /tank/wk/TSV2_paper_clean_v4
```

复用数据和权重：

```bash
ln -sfn /tank/wk/TSV2/data data
ln -sfn /tank/wk/TSV2/text_features_llm text_features_llm
ln -sfn /tank/wk/TSV2/pretrained pretrained
```

正式 TransUNet 主表使用 ImageNet-21k 的 `R50+ViT-B_16.npz`。验证或准备权重：

```bash
# 已经有文件时
bash scripts/prepare_transunet_pretrained.sh \
  /absolute/path/R50+ViT-B_16.npz

# 或直接放到下面的位置后验证
python scripts/verify_transunet_pretrained.py \
  pretrained/transunet/R50+ViT-B_16.npz
```

默认计划中的 `transunet_pretrained` 会强制加载该文件；缺失或格式不正确时
`python scripts/paper.py validate` 会失败，绝不会静默退回随机初始化。

本包已包含上传时提供的四套官方源码：

```text
third_party/LViT
third_party/SAM-Med2D
third_party/Swin-Unet
third_party/TransUNet
```

## 4. 启动前检查

```bash
conda activate TS
python scripts/paper.py validate
```

检查内容包括：六域数据、文本特征、本地 CLIPSeg、BERT、SAM-Med2D 权重、官方源码、Conda 环境和六张 GPU。

生成并查看全部命令：

```bash
python scripts/paper.py plan --show-commands
```

所有解析后的命令同时写入：

```text
paper_artifacts/plan.tsv
paper_artifacts/plan.json
```

## 5. 一条命令启动 33 次实验

```bash
python scripts/paper.py launch --fresh
```

该命令会为每张 GPU 启动一个 SSH-safe 后台 worker。每张卡串行执行自己的队列；一个任务失败不会让其他 GPU 的队列停止。

查看状态：

```bash
python scripts/paper.py status
watch -n 2 nvidia-smi -i 0,1,2,4,5,6
```

调度日志：

```text
paper_artifacts/scheduler/gpu0.log
paper_artifacts/scheduler/gpu1.log
...
```

每个 run：

```text
paper_artifacts/runs/<experiment>/seed_<seed>/
├── resolved_run.json
├── train.log
├── predict.log
├── checkpoint.pth
├── checkpoint.pth.history.csv
├── predictions/
└── completed.json 或 FAILED.txt
```

## 6. 精简日志

后台 worker 自动设置：

```text
PAPERLAB_PROGRESS=off
```

因此 tqdm 不再把 1500 个 iteration 写成 1500 行。日志只保留：

- 启动配置；
- 每个 epoch 的训练/验证汇总；
- 必要的六域指标；
- AMP overflow、NaN、early stop 和异常。

直接在交互终端手工运行训练脚本时，默认仍显示单行动态进度条。也可强制关闭：

```bash
PAPERLAB_PROGRESS=off python scripts/train.py ...
```

强制显示：

```bash
PAPERLAB_PROGRESS=on python scripts/train.py ...
```

## 7. 唯一权威结果文件

所有训练历史和预测指标自动汇总到：

```text
paper_artifacts/experiment_results.sqlite
```

SQLite 包含：

- `project_metadata`：配置、代码和系统信息；
- `runs`：模型、种子、GPU、batch、命令和状态；
- `epochs`：逐 epoch 训练与验证指标；
- `case_metrics`：逐测试病例 Dice、IoU、HD95；
- `domain_metrics`：逐域、macro、weighted 指标；
- `paired_statistics`：配对统计结果。

每个预测任务完成后立即写入数据库，不需要等待全部训练结束。

从已有 run 文件夹重建数据库：

```bash
python scripts/paper.py collect
```

## 8. 导出论文表格

```bash
python scripts/paper.py export
```

导出：

```text
paper_runs.csv
paper_epochs.csv
paper_case_metrics.csv
paper_domain_metrics.csv
paper_results_long.csv
paper_run_summary.csv
paper_method_summary.csv
```

配对病例级统计：

```bash
python scripts/paper.py stats --reference tasksegment_full
```

## 9. 恢复与重跑

默认行为：已有 `completed.json` 的 run 会跳过；未完成或失败的 run 会重新进入队列。

重建计划并保留已有结果：

```bash
python scripts/paper.py launch
```

强制重跑选定实验：

```bash
python scripts/paper.py launch \
  --experiments tasksegment_full \
  --seeds 42 \
  --force
```

完全清空正式输出并从头开始：

```bash
python scripts/paper.py launch --fresh
```

`--fresh` 只删除当前工程的 `paper_artifacts/`，不会删除数据、预训练权重或旧工程。

## 10. 论文协议说明

- TaskSegment 的 Full 与三个消融在同一 seed 下固定到同一张 RTX 3090，以减少硬件混杂。
- LViT 使用 native 224 输入，因为 512/batch-1 已确认存在严重 BatchNorm running-statistics 失配；其 logits 在阈值化前上采样到 512。
- SAM-Med2D 主表只使用 prompt-free 自动分割。GT box/point/mask 不进入自动主表。
- CLIPSeg 使用本地完整模型目录，后台强制离线加载。
- 所有方法使用相同数据划分、测试病例、阈值 0.5、无测试后处理和 512 指标实现。
- 若某模型在 RTX 3090 上以真实 batch 4 OOM，任务会明确失败；调度器不会静默降低 batch。应记录后再决定是否使用 activation checkpointing，而不是偷偷改变协议。

详见：

- `docs/PAPER_PROTOCOL_CN.md`
- `docs/RESULT_DATABASE_CN.md`
- `docs/MIGRATION_CN.md`
