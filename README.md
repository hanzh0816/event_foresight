# Event Foresight — DINO-world baseline

本分支以 [DINO-world: Back to the Features](https://arxiv.org/abs/2507.19468) 为主要 RGB baseline，在 DSEC 上预测未来目标检测。分支 `codex/dino-world-baseline` 从 `558ed53` 创建；该提交包含独立 RGB baseline、DSEC 数据检查和现有评测流程。

```text
四张历史 RGB → 冻结 DINOv2 ViT-B/14-reg → 768 维最后层 patch features
             → DINO-world causal cross-attention predictor → 未来 features
             → 新训练的 Deformable DETR → 原图坐标的检测框、类别和分数
```

输入仍为间隔约 50ms 的四张 640×480 历史图像，默认 resize 至 448×896，预测未来 50ms。DSEC 读取、40 帧 sample inventory、train/val/test split、标注处理、COCO AP/AR、样本配对与预测导出逻辑保持不变。

截至 2026-11-26 的来源核查未找到作者公开的 DINO-world predictor 代码或预训练权重。本分支根据论文实现 predictor，在 DSEC 上从随机初始化训练；encoder 使用官方预训练权重。这是 DINO-world 方法的 DSEC 实现，尚无训练结果，不代表论文大规模视频预训练模型的复现结果。

## 模型与监督

- Encoder：冻结 `dinov2_vitb14_reg`；逐帧提取最后一个 block 经最终 LayerNorm 后的 patch tokens，丢弃 CLS/register tokens。不使用多层拼接、PCA 或额外 L2 normalization。默认网格为 32×64，每个 token 为 768 维。
- Predictor：论文 Base 尺寸，12 层、宽度 768、12 heads、MLP ratio 4；每个 query 从同一个可学习向量初始化，各层为 pre-LN cross-attention 与 residual MLP，末尾 LayerNorm 和线性输出。没有 query self-attention，context 不随层更新。
- 位置编码：每个 head 的 60 维分成时间、纵坐标、横坐标各 20 维，剩余 4 维不旋转；空间坐标使用 `[-1,1]` 网格，时间使用秒。
- 因果监督：历史 `z0,z1,z2,z3` 对应 query 时刻 `t1,t2,t3,t4`；各 query 只读 `t_context < t_query` 的 context。targets 为 `z1,z2,z3,z4`，其中 `z4` 由独立的未来 RGB 编码得到，全部 stop-gradient。训练平均所有四步、空间位置和 feature 维度上的 `SmoothL1(beta=0.1)`。
- 推理：检测只用四张历史 RGB，返回 `t4` 特征。`predict_at` 支持显式时间，`rollout` 支持潜特征自回归反馈；当前 DSEC AP 评测仍只测 +50ms，未新增多步评测。

论文没有公开全部工程细节。本实现明确采用 PyTorch Linear 默认初始化、query `Normal(0,0.02)`、带 bias 的 Q/K/V/output projections、GELU、LayerNorm `eps=1e-5`、无 LayerScale、默认 dropout=0、AdamW betas `(0.9,0.999)`。RoPE 将论文的 `10**linspace(-2,2,10)` 解释为周期，按 `2*pi*coordinate/period` 旋转相邻二维对；空间网格取端点，单点维度取 0。上述选择不承诺与未发布的官方 checkpoint 兼容。

DSEC 适配沿用四帧、名义 50ms 时间间隔与原 resize，训练序列合计五帧；论文的大规模预训练使用八帧、变帧率及不同分辨率阶段。这里没有更改 DSEC reader 来引入 variable-FPS。自回归 API 使用固定长度滑动 context。默认按 1024 个 query 分块计算 attention，并在每层只投影一次 context K/V；可启用 activation checkpointing。Base 全局时空 attention 的实际显存需求仍需在目标 GPU 上验证。

## 安装与手动下载权重

```bash
python -m pip install -e '.[test]'
```

只需要下载一份官方 **encoder** 权重，约 330 MiB：

- 来源：[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- 文件：[dinov2_vitb14_reg4_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)
- 建议存放位置：`/data3/hzh/models/dino_world/dinov2_vitb14_reg4_pretrain.pth`

在需要运行模型的服务器上手动执行：

```bash
mkdir -p /data3/hzh/models/dino_world
curl -fL --retry 3 -C - \
  'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth' \
  -o /data3/hzh/models/dino_world/dinov2_vitb14_reg4_pretrain.pth
python scripts/link_checkpoints.py --root /data3/hzh/models/dino_world
```

脚本将项目内 `checkpoints` 链接至该目录；不会覆盖已有目录或其他软链接。如果已有 `checkpoints`，直接将 YAML 中的 `model.encoder_checkpoint` 设置为下载文件的绝对路径。构建模型时使用 `weights_only=True` 严格加载官方原始 state_dict，不自动下载、不调用 torch.hub。模型入口支持 PyTorch 新旧序列化格式。

DINO-world predictor 与 768 维检测头均需后续训练生成 checkpoint。旧 `dinof.pt`、1152 维 PCA predictor 和 3072 维 `detector.pt` 无法用于新路径。`scripts/import_weights.py` 仅保留旧 DINO-Foresight 权重转换用途。

## 后续训练入口

以下是准备好的运行命令；本次代码变更没有启动 DSEC 训练，也没有下载权重。先修改 YAML 的 `data.root`。正式训练要求 CUDA。

第一阶段训练 predictor：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --stage predictor --config configs/predictor.yaml \
  --output-dir outputs/dino_world_predictor
```

默认冻结 encoder；优化 predictor 的所有参数。配置采用 `lr=1e-4`、`weight_decay=0.4`、5000 步线性 warmup 后恒定学习率；DSEC 训练预算默认 40 epochs、无 step 截断，可按实验预算调整。每卡 batch 为 1、梯度累积为 8。保存 `last.ckpt`，并按验证集 next-frame feature loss 保存 `best.ckpt`；该 loss 选出的模型仍需通过检测 AP 评价。

第二阶段固定已训练的 predictor，重新训练检测头：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --stage detector --config configs/detector.yaml \
  --predictor-checkpoint outputs/dino_world_predictor/checkpoints/best.ckpt \
  --output-dir outputs/dino_world_detector
```

检测头沿用原 Deformable DETR 架构与检测损失，输入维度改为 768。未提供已训练 predictor 时，检测头训练会直接报错。`data.batch_size` 为每卡 batch，有效 batch = 每卡 batch × GPU 数 × `accumulation_steps`；自动使用所有可见 GPU。

同阶段断点恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train --stage predictor \
  --config configs/predictor.yaml \
  --resume outputs/dino_world_predictor/checkpoints/last.ckpt
```

完整 checkpoint 保存 encoder、predictor、检测头（若有）、optimizer、scheduler、训练步数及模型/数据协议。`--resume` 恢复原训练配置，允许数据和 encoder 文件搬迁，要求原 batch、累积、GPU 数与样本选择一致。epoch 中途恢复可能重读该轮开头的样本。DINO-world 使用新的 checkpoint format，旧 DINO-Foresight 训练 checkpoint 会被拒绝。

## 推理与评测

推理需要完整的本项目 DINO-world Detector checkpoint；恢复其中的 predictor 后，可用显式 `--predictor-checkpoint` 覆盖。默认配置的 predictor 路径为 `null`，因此完整 Detector checkpoint 不依赖此前 predictor 文件的位置。

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.evaluate \
  --config configs/detector.yaml --split val --data-root /data3/hzh/DSEC \
  --checkpoint outputs/dino_world_detector/checkpoints/best.ckpt \
  --output outputs/dino_world_val.json

CUDA_VISIBLE_DEVICES=0 python -m foresight.infer \
  --checkpoint outputs/dino_world_detector/checkpoints/best.ckpt \
  --images frame0.png frame1.png frame2.png frame3.png \
  --output outputs/dino_world_prediction.json
```

支持原有 `--manifest configs/val600.json`、`--max-samples` 和 `--split test`。指标 JSON 仍使用 0–1 数值，并导出各样本预测。预测框使用原图 `640×480` 坐标，类别和 AP/AR 计算不变。

## 数据检查与开发验证

```bash
python -m pip install -e '.[event]'
python scripts/check_dsec.py --data-root /path/to/DSEC
# 仅检查 RGB：
python scripts/check_dsec.py --data-root /path/to/DSEC --rgb-only
python -m pytest -q
```

CPU 单元测试检查 3D RoPE、各时间边界的因果性、teacher-forcing loss、冻结 encoder、分块与 activation checkpointing 的输出/梯度、完整训练状态恢复以及原有数据/评测合同。测试使用微型模型和合成数据；真实 encoder 权重加载、CUDA/BF16 与 DSEC AP 仍需在权重和数据就绪后验证。

相关文献见 [Event 表征调研](docs/event-representation-survey.md)。保留的旧模型/vendor 代码用于旧资产转换及回归测试；默认配置、训练、推理和评测入口已切换至 DINO-world。来源与许可证见 [SOURCES.json](foresight/vendor/SOURCES.json) 和 [检测模块许可证](foresight/models/APACHE-LICENSE)。权重与 outputs 不提交到 Git。
