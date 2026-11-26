# Event Foresight v3.1

用历史 RGB 与事件预测 DSEC 下一时刻的 DINOv2 特征，再用共享冻结检测头评测未来目标检测。默认间隔 50 ms，四帧历史；模型由冻结 DINOv2、RVT-S 和 8 层双路 Cross-Attention 预测器组成。

实现以项目 v3.1 设计为准。架构、监督、消融与服务器验收说明见 [docs/v31.md](docs/v31.md)。最终集成目前仅完成静态源码检查，尚未进行集成运行或 DSEC 训练评测。

## 安装、权重与数据

以下命令在训练服务器执行：

```bash
python -m pip install -e '.[event,test]'
mkdir -p /data3/hzh/models/dino_world
curl -fL --retry 3 -C - \
  'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth' \
  -o /data3/hzh/models/dino_world/dinov2_vitb14_reg4_pretrain.pth
```

已有该权重时无需重新下载。模型严格加载本地官方 `dinov2_vitb14_reg4_pretrain.pth`；不自动下载。编码器去掉 CLS/register tokens，使用最后层经 LayerNorm 的 768 维 patch 特征，无 PCA。

在 `configs/predictor.yaml` 和 `configs/detector.yaml` 设置 `data.root`、`model.encoder_checkpoint`。事件配置继承 RGB predictor 配置。数据需要 DSEC detection 已映射到左事件相机坐标的 `images/left/distorted`、`images/timestamps.txt`、`object_detections/left/tracks.npy`，以及 `events/left/events.h5`。具体坐标与目录约定见 [输入数据](docs/v31.md#输入数据与时间边界)。

## 训练与评测

1. 先用真实 RGB 的冻结 DINO 特征训练共享检测头：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --stage detector --config configs/detector.yaml \
  --output-dir outputs/v31_detector
```

检测头训练不需要 predictor checkpoint。训练完成后，在 `configs/predictor.yaml` 设置 `model.detector_checkpoint: outputs/v31_detector/checkpoints/best.ckpt`，所有预测器共用该冻结头。

2. 训练完整 Event-RGB 模型 M2：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --stage predictor --config configs/event_predictor.yaml \
  --output-dir outputs/v31_m2
```

将配置换为 `configs/predictor.yaml` 或 `configs/event_no_align.yaml` 可分别训练 RGB 基线 B1、无对齐辅助的 M1，并使用独立输出目录。默认每卡 batch 1、累积 8 步、40 epochs；预算和显存需在服务器核查。提供共享头时 `best.ckpt` 按 val AP 选择，`best_foreground.ckpt` 按前景 latent L1 选择，`last.ckpt` 用于恢复。

3. 用同一个共享头评测预测器：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.evaluate \
  --config configs/event_predictor.yaml --split val \
  --checkpoint outputs/v31_detector/checkpoints/best.ckpt \
  --predictor-checkpoint outputs/v31_m2/checkpoints/best.ckpt \
  --event-mode correct --horizon-steps 1 \
  --output outputs/v31_m2_val.json
```

评测输出 COCO AP/AR、类别 AP、分区域 latent 指标、样本清单及预测框。AP 使用 0 到 1 的数值，框坐标对应原图 `640×480`。`--event-mode`、多步 rollout、高动态子集和独立图片推理的用法见 [评测与推理](docs/v31.md#评测与推理)。

## 文件入口

- [模型与服务器交接说明](docs/v31.md)：结构、配置、因果边界、恢复与待验证项目。
- [Event 表征调研](docs/event-representation-survey.md)：相关文献。
- [上游来源](foresight/vendor/SOURCES.json)、[检测模块许可证](foresight/models/APACHE-LICENSE)：保留的第三方代码来源与许可。

权重、数据和 `outputs/` 不提交到 Git。
