# Event Foresight v3.1

用历史 RGB 与事件预测 DSEC 下一时刻的 DINOv2 特征，再用共享冻结检测头评测未来目标检测。默认间隔 50 ms，四帧历史；模型由冻结 DINOv2、RVT-S 和 8 层双路 Cross-Attention 预测器组成。B1/M1/M2 可以直接开始自监督训练，无需先训练或加载检测头。

实现以项目 v3.1 设计为准。架构、监督、消融与验证记录见 [docs/v31.md](docs/v31.md)。2026-11-27 已在 Delta 完成无训练检查，包括真实 DSEC 的 batch 1/3 BF16 前向、反向与因果性验证；尚未启动训练或评测 AP。

## 安装、权重与数据

在每台训练服务器设置 `Work`，指向存放数据、权重和训练产物的根目录。变量名区分大小写；以下示例沿用 Delta 的数据盘位置：

```bash
export Work=/data3/hzh
python -m pip install -e '.[event,test]'
mkdir -p "$Work/models/dino_world"
curl -fL --retry 3 -C - \
  'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth' \
  -o "$Work/models/dino_world/dinov2_vitb14_reg4_pretrain.pth"
```

已有该权重时无需重新下载。模型严格加载本地官方 `dinov2_vitb14_reg4_pretrain.pth`；不自动下载。编码器去掉 CLS/register tokens，使用最后层经 LayerNorm 的 768 维 patch 特征，无 PCA。

配置统一使用 `$Work/DSEC`、`$Work/models/` 和 `$Work/outputs/`。保持这组子目录布局时，切换服务器只需修改 `Work`；目录不同则调整 YAML 中对应后缀。事件配置继承 RGB predictor 配置。数据需要 DSEC detection 已映射到左事件相机坐标的 `images/left/distorted`、`images/timestamps.txt`、`object_detections/left/tracks.npy`，以及 `events/left/events.h5`。具体坐标与目录约定见 [输入数据](docs/v31.md#输入数据与时间边界)。

默认配置要求 `Work` 是非空绝对路径，未设置时会报错。YAML 和命令行中的运行路径支持 `$Work`、`${Work}`；不带根目录的相对路径也从 `Work` 解析，绝对路径可直接使用。`--config` 仍相对代码仓库，`extends` 相对当前 YAML，详细规则见 [配置与训练预算](docs/v31.md#配置与训练预算)。旧仓库中的 `outputs/` 不会自动迁移。

## 训练与评测

在仓库目录运行。完整 Event-RGB 模型 M2 直接启动：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --config configs/event_predictor.yaml
```

M2 联合训练 RVT、Event neck 和 predictor，DINOv2 冻结；监督来自未来特征预测和历史 Event-RGB 对齐损失。默认 `model.detector_checkpoint: null`，不加载检测头、不计算 AP。`best.ckpt` 按验证预测损失 `val_feature_loss` 选择，`best_foreground.ckpt` 按前景 latent L1 选择，`last.ckpt` 用于恢复。

RGB 基线 B1 和无对齐辅助的消融 M1 分别启动：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --config configs/predictor.yaml

CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --config configs/event_no_align.yaml
```

三组相互独立，M2 不需要 B1 权重。配置分别保存到 `$Work/outputs/v31_b1`、`$Work/outputs/v31_m1`、`$Work/outputs/v31_m2`。默认每卡 batch 1、累积 8 步、40 epochs。当前继续使用原来目标帧有有效检测框的样本集合，数据加载仍需 `tracks.npy`；检测框用于筛选窗口和统计前景指标，不进入 predictor 的训练 loss。

共享检测头使用真实 RGB 的冻结 DINO 特征和检测标注独立训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.train \
  --config configs/detector.yaml
```

默认每卡 batch 4、累积 1 步、40 epochs，最佳权重保存为 `$Work/outputs/v31_detector/checkpoints/best.ckpt`。它与 predictor 训练相互独立，可分配不同 GPU 同时训练。上面的 `0` 为卡号示例，需替换为实际分配的 GPU。

实验设置写在 YAML 中：`data` 管理 batch 和 workers，`optimization` 管理学习率、累积和训练预算，`model` 管理结构及权重路径。训练命令只接受 `--config`、`--resume` 和可选的 `--output-dir`。恢复默认沿用 checkpoint 配方；字段说明与恢复方式见 [配置与训练预算](docs/v31.md#配置与训练预算)。

观察训练时，在 TensorBoard 的 Scalars 中筛选 `^monitor/`，或打开 Custom Scalars 的 `Training overview`。其余指标归入 `details/`；完整 CSV 和逐轮验证报告继续保留。启动命令与指标含义见 [训练日志](docs/v31.md#训练日志)。

predictor 与检测头都完成后，用同一个冻结头评测：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.evaluate \
  --config configs/evaluate_event.yaml
```

`configs/evaluate_event.yaml` 已指定 M2 predictor 权重、共享检测头和结果路径；B1 使用 `configs/evaluate_rgb.yaml`。调整 YAML 的 `evaluation` 部分可选择 split、事件干预、预测步数和子集，临时结果路径可用 `--output` 覆盖。需要在 predictor 训练期间验证 AP 时，见 [可选 AP 验证](docs/v31.md#共享检测头与-checkpoint)。

评测输出 COCO AP/AR、类别 AP、分区域 latent 指标、样本清单及预测框。AP 使用 0 到 1 的数值，框坐标对应原图 `640×480`。多步 rollout、高动态子集和独立图片推理的用法见 [评测与推理](docs/v31.md#评测与推理)。

## 文件入口

- [模型与服务器交接说明](docs/v31.md)：结构、配置、因果边界、恢复与待验证项目；模块位置见[代码结构](docs/v31.md#代码结构与来源)。
- [Event 表征调研](docs/event-representation-survey.md)：相关文献。
- [上游来源](foresight/vendor/SOURCES.json)、[检测模块许可证](foresight/models/detection/APACHE-LICENSE)：保留的第三方代码来源与许可。

权重、数据和 `$Work/outputs/` 不提交到 Git。
