# Iris-2D：本机 TSV3 核对后的服务器更新包

本包取代上一版 iris2d_baseline.zip。以 D:/task learning+segment/TSV3/TaskSegment 的实际代码为准，
Iris 的基准学习率已从上一版 5e-5 更正为 **1e-4**。

本机源代码根目录为 TSV3/TaskSegment，数据位于外层 TSV3/data。
服务器代码根目录为 /tank/wk/TS/TSV3；上传时将本包内容解压到这个根目录，
不要额外套一层 TaskSegment。

## 1. 与其他对比实验保持一致的设置

| 设置 | Iris-2D 当前值 |
|---|---|
| 优化器 | AdamW，默认 betas=(0.9,0.999)，eps=1e-8 |
| 基准/峰值学习率 | 1e-4 |
| 最低学习率 | 1e-6 |
| weight decay | 1e-4 |
| 预热 | 5 epochs，线性增长 |
| 衰减 | 按 epoch 余弦下降，固定 150-epoch 计划 |
| 查询 batch / 梯度累积 | 4 / 4，有效查询 batch=16 |
| 每 epoch | 1,500 个域均衡训练步骤 |
| 上限 / early stopping | 150 epochs / 验证宏平均 Dice 连续 20 epochs 未提高 |
| 梯度裁剪 | 5.0 |
| 种子 | 42、3407、2026 |
| 评价 | 512×512、阈值 0.5、无形态学后处理 |

学习率曲线与本机 U-Net、U-Net++、Swin-Unet、TransUNet 的实现逐 epoch 相同。
第 1–5 个 epoch 为 2e-5、4e-5、6e-5、8e-5、1e-4，第 6 个仍为 1e-4，
第 150 个为 1e-6。提前停止不会压缩这条固定计划。

SAM-Med2D、LViT、CLIPSeg 的预热实现包含最低学习率偏移，起点略有不同；
本包以四个图像基线的共同曲线为准。详见 IRIS2D_TSV3_AUDIT.md。
原有对比实验、数据、权重和历史结果均未改动。

训练预算自动读取现有 configs/paper_3seeds.json；新增 configs/iris2d_baseline.json
保存 Iris 的模型与学习率设置。正式运行会拒绝通过命令行将学习率、训练预算改成不同值。
源码和协议检查忽略 CRLF/LF 差异，避免 Windows 上传 Linux 后出现行尾误报。

## 2. 模型的身份与范围

名称建议使用 **Iris-2D (paper-based adaptation)**，属于根据 Gao 等 CVPR 2025
Show and Segment 论文编写的独立二维适配，不是作者官方实现或原论文精确复现。

保留共享图像编码器、高分辨率前景池化、全分辨率 PixelShuffle/掩码融合/PixelUnshuffle、
上下文 cross/self attention、双向任务/查询注意力及任务 token 驱动的掩码头。
不使用文本、域 embedding、文本稠密引导或原型 EMA。

实现选择：二维 U-Net 通道 32/64/128/256/512、GroupNorm+GELU、下采样率16，
10个上下文 tokens 加1个前景 token、8头注意力、2层双向解码。
默认参数量 24,951,912。训练用1个支持样本和4个查询，评价固定4个训练支持样本集成。
损失保留 Iris 的 CE+Dice（权重各1），没有改成 US-TextSeg 的 Focal-Tversky 损失。
本包统一的是当前超声对比实验的学习率与预算，不是原论文的3D/LAMB训练方案。

论文：
https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Show_and_Segment_Universal_Medical_Image_Segmentation_via_In-Context_Learning_CVPR_2025_paper.html

## 3. 上传、核对、短检查

将 iris2d_tsv3_update.zip 上传到服务器源码根目录，然后：

```bash
cd /tank/wk/TS/TSV3
conda activate TS
unzip -o iris2d_tsv3_update.zip -d .
python scripts/preflight_iris2d.py --check-data
python scripts/check_iris2d.py
```

预检会比较本机审阅版本的数据/增强/指标函数、四个基线的学习率函数及公共训练协议。
如果提示 Source/protocol differs，请先核对列出的服务器文件，不要跳过检查。
本包没有自动覆盖这些既有核心文件。数据检查核对图像/掩码配对数量，
不提供患者级独立性或不同文件内容无重复的证明。

在空闲 GPU 上先跑1个训练步骤和每域2个验证样本，将0换成空闲卡号：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_iris2d.py \
  --seed 42 --smoke --cache-dataset \
  --output paper_artifacts/runs/iris2d_smoke_local_aligned
```

默认仍为512×512和完整模型，可检查实际显存、AMP与数据接口。
smoke输出会被三种子汇总脚本拒绝，不可填入论文结果。

## 4. 正式训练与测试

先查看将执行的命令，不启动训练：

```bash
python scripts/launch_iris2d.py --gpu 0 --dry-run
```

三个种子依次训练、测试并汇总：

```bash
python -u scripts/launch_iris2d.py --gpu 0
```

仅运行一个种子：

```bash
python -u scripts/launch_iris2d.py --gpu 0 --seeds 42
```

也可使用 bash scripts/run_iris2d_3seeds.sh 0。
物理GPU由--gpu指定，各子进程使用映射后的cuda:0。
默认数据位置同时支持本机的外层data和服务器同级data；需要时显式传--data-root。
--run-root可指定结果目录。脚本独立运行，不会启动已有paperlab里的其他实验。

断点续训仅选择未完成的种子，例如：

```bash
python -u scripts/launch_iris2d.py --gpu 0 --seeds 42 --resume
```

恢复自该种子 checkpoint.last.pth，保持学习率所在epoch、优化器、scaler和随机状态。
已完成的测试目录不会自动覆盖。中断发生在epoch中间时，重跑该epoch。

## 5. 结果

默认位于 paper_artifacts/runs/iris2d/seed_42，其他种子同理：

- checkpoint.pth：验证宏平均Dice最优的权重。
- checkpoint.last.pth：断点续训状态。
- checkpoint.pth.config.json：与已有实验相同风格的训练参数记录。
- checkpoint.pth.history.csv：逐epoch学习率、训练损失、验证Dice/IoU/HD95。
- protocol.json、splits.json、support_manifest.json：配置和数据/支持清单。
- predictions/metrics.json、per_case.csv：正式测试汇总和逐图指标。
- 三种子 summary_k4.json：跨种子的均值与样本标准差。

模型和数据清单只根据训练集/验证集选择，不用测试集选择checkpoint、阈值或支持样本。
源码指纹与数据清单在测试和续训时再次核对。

如需额外单支持测试，仍使用同一个已选checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict_iris2d.py \
  --checkpoint paper_artifacts/runs/iris2d/seed_42/checkpoint.pth \
  --support-count 1 --output paper_artifacts/runs/iris2d/seed_42/predictions_k1
```

## 6. 环境与验证

沿用服务器 TS 环境。依赖已满足时无需安装或升级。
确实缺包时参考 requirements-iris2d.txt。
本地检查范围与尚未验证的GPU环境详见 VALIDATION.md。
正式超声训练和三种子结果尚未生成。

## Python版本兼容修复

已修正 Python3.10/3.12 的 AST 签名误报；注释/空白差异采用完整 AST 等价检查。
详见 IRIS2D_COMPAT_FIX.md。若预检仍提示实际逻辑不同，运行 scripts/diagnose_iris2d.py 收集参考源码。
