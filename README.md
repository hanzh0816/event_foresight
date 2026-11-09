# Event Foresight

基于 DINO-Foresight 的 RGB/Event 潜在预测与未来目标检测。Event 分支支持当前 RGB 特征对齐，以及通过冻结 Predictor 回传未来监督来训练 Event Encoder。

输入为四张按时间排序、间隔约 50ms 的 640×480 图片，预处理至 448×896；预测未来 50ms 的目标，输出原图坐标下的检测框、类别和分数。

## 安装与权重

```bash
git clone --branch codex/event-future-supervision https://github.com/hanzh0816/event_foresight.git
cd event_foresight
python -m pip install -e '.[event,test]'
python scripts/link_checkpoints.py --root /path/to/model-assets
```

运行需要支持 BF16 的 CUDA GPU。先安装适合目标机器的 CUDA PyTorch，再安装本项目；已测试环境为 Python 3.12、PyTorch 2.14.1、Lightning 2.6.6。模型源码已随仓库收录，运行不依赖旧项目，也不会自动联网下载权重。

权重目录包含 `dinof.pt`、`detector.pt` 和 `assets.json`；`checkpoints` 通过软链接访问它们。也可以把实际文件直接放在项目的 `checkpoints/` 下。建立链接时使用目标机器的目录：

```bash
python scripts/link_checkpoints.py --root /path/to/checkpoints
```

DSEC 数据根目录在 `configs/*.yaml` 的 `data.root` 中设置。配置内的相对路径以项目根目录解析，输出默认保存到 `outputs/`。

| 分支 | 内容 |
|---|---|
| `main` | RGB 基线 |
| `codex/event-encoder` | Event Encoder 当前特征对齐 |
| `codex/event-temporal-sampling` | 50ms/200ms 均匀与非均匀分箱 |
| `codex/event-future-supervision` | 包含前述功能，以及当前/未来联合训练、全量配置和双头评测 |

## DSEC 数据检查

完整 RGB/Event 检查需要额外安装 HDF5 读取依赖：`python -m pip install -e '.[event]'`。

搬运数据后，先在安装了本项目依赖的 Python 环境中检查完整 train/val/test：

```bash
python scripts/check_dsec.py --data-root /path/to/DSEC
# 可选：保存 JSON 报告（路径放在 DSEC 之外）
python scripts/check_dsec.py --data-root /path/to/DSEC --output outputs/dsec_check.json
```

默认检查 RGB + Event：逐序列检查文件、所有 RGB 帧是否存在且非空、时间戳、检测标注、曝光时间、HDF5 结构及全部有效窗口的事件覆盖，并抽样解码图片和事件。使用项目默认 40 帧历史、4 帧输入、50ms 预测协议，检查固定的 41/6/12 个序列，不读取子集 manifest。输出 `train/eval/test=PASS/FAIL` 和有效窗口数；训练检查包含 train 和 val，eval 对应 val。通过退出码为 `0`，数据缺失或无效为 `1`。

追加 `--deep` 可解码全部 PNG、分块扫描全部事件并校验时间索引，适合检查传输损坏；耗时取决于数据量。200ms 实验追加 `--event-window-us 200000`；仅跑 RGB 时用 `--rgb-only`。Event 检查要求曝光文件齐全，以保持曝光结束时刻对齐协议。脚本只检查数据，不加载模型权重或使用 GPU。

## 推理与评估

四图推理不需要标签或未来图像：

```bash
CUDA_VISIBLE_DEVICES=3 python -m foresight.infer \
  --images frame0.png frame1.png frame2.png frame3.png \
  --output outputs/prediction.json
```

评估默认使用官方 Predictor 和已有 RGB Detector：

```bash
CUDA_VISIBLE_DEVICES=3 python -m foresight.evaluate \
  --split val --data-root /data3/hzh/DSEC \
  --output outputs/rgb_val.json
```

可用 `--manifest configs/val600.json` 评估固定子集，或用 `--max-samples 8` 快速检查。指标 JSON 使用 0–1 数值，同时导出各样本预测。

评估和推理均支持 `--checkpoint <Detector checkpoint>`；替换 Predictor 时使用 `--predictor-checkpoint <Predictor checkpoint>`。

## 训练

默认冻结 Encoder/PCA/Predictor，仅训练检测头。已有 RGB Detector 用于初始化；从头训练时将 `model.detector_checkpoint` 设为 `null`。

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train \
  --stage detector --config configs/detector.yaml
```

可选的 Predictor 训练冻结 Encoder/PCA，以真实未来 RGB 特征为目标：

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train \
  --stage predictor --config configs/predictor.yaml
```

训练自动使用全部可见 GPU。`data.batch_size` 为每卡 batch，有效 batch 为每卡 batch × GPU 数 × `accumulation_steps`。日志、配置和 checkpoint 保存到 `outputs/<stage>_<time>/`。

恢复同阶段训练：

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train --stage detector \
  --resume outputs/detector_RUN/checkpoints/last.ckpt
```

`--resume` 恢复模型、优化器、调度器和训练步数，不接受导入的 `dinof.pt`/`detector.pt`。保持原 GPU 数和 batch 设置；epoch 中途恢复可能重读该轮开头的样本。

使用训练后的 Predictor 训练检测头时，在配置中设置 `model.predictor_checkpoint`。

## 在另一台服务器运行全量 Event 实验

使用 `codex/event-future-supervision` 分支。两组都直接从 `checkpoints/dinof.pt` 中的 RGB 预训练 Encoder 初始化，采用相同的 10 通道输入层初始化；训练全部 Event Encoder 参数，冻结 RGB Teacher、PCA 和 Predictor。

| 配置 | 当前监督 | 未来监督 | 数据与预算 |
|---|---|---|---|
| `configs/event_full_current.yaml` | 四个历史状态 | 无 | 全 train，10 epochs |
| `configs/event_full_joint.yaml` | 四个历史状态 | 真实未来 RGB 特征 | 相同 |

两组均使用 50ms 均匀分箱、学习率 `1e-4`、每卡 batch 1、累积 4 次、seed 42。未来监督权重分别为 0 和 1，当前监督权重均为 1。`max_steps: -1` 按完整 epoch 训练；可用 `--epochs` 和 `--seed` 设置成对实验的共同预算与种子。完整训练/验证/测试窗口分别为 **46,719 / 7,806 / 9,587**，配置不使用子集 manifest。

### 数据和权重

数据保留以下目录；验证序列也位于物理 `train/` 目录，具体序列划分见 `foresight/data/definitions.py`。

```text
DSEC/
  train/<sequence>/
    images/left/distorted/000000.png ...
    images/timestamps.txt
    images/left/exposure_timestamps.txt
    events/left/events.h5
    object_detections/left/tracks.npy
  test/<sequence>/
    ...同样结构...
```

训练需要事件流、历史及未来 RGB，以及用于生成一致窗口清单的检测标注。设置 `--data-root /path/to/DSEC` 即可迁移数据路径。

Git 之外准备这些资产：

```text
checkpoints/
  dinof.pt                         # 训练必需：RGB Teacher、PCA、Predictor
  detector.pt                      # 冻结检测头，用于评测
  assets.json                      # 权重哈希与来源
  readouts/rgb_current/
    roundtrip_last.pt               # 第二个固定检测头，用于双头评测
    plan.json                      # 与第二个头保持原字节，加载时校验哈希
  event/
    e2_7500.pt                     # 旧实验复测及 pilot 初始化
    200ms_recent_fine.pt            # 旧200ms实验复测
```

全量两组不需要 `event/` 中的权重。模型文件为本项目转换后的格式；有原始官方/检测头 checkpoint 时，可使用 `scripts/import_weights.py --help` 转换。

已准备的搬运包名为 `event_foresight_assets.tar`（约2.35GB），配套 `.sha256`；它们与代码分开传输。在新 checkout 根目录校验并解压，即可得到上述 `checkpoints/`，不需要再运行链接脚本：

```bash
sha256sum -c event_foresight_assets.tar.sha256
tar -xf event_foresight_assets.tar
python -m pytest -q
FORESIGHT_ASSET_DIR="$PWD/checkpoints" python -m pytest -q tests/test_models.py
```

### 成对训练

以下示例每组使用两张 GPU，有效 batch 为 8。两组使用相同 GPU 数与训练设置，可顺序运行，也可分配到两组空闲 GPU。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m foresight.train_future_event \
  --config configs/event_full_current.yaml --data-root /path/to/DSEC \
  --output-dir outputs/full_current_seed42

CUDA_VISIBLE_DEVICES=0,1 python -m foresight.train_future_event \
  --config configs/event_full_joint.yaml --data-root /path/to/DSEC \
  --output-dir outputs/full_joint_seed42
```

先检查少量数据时，为两组各使用单独输出目录，追加 `--max-steps 2 --limit-train-batches 8 --limit-val-batches 2`。正式训练使用上面的完整命令。

运行会保存解析后的配置、数据清单计数与哈希、梯度审计和训练日志。每个 epoch 保留 checkpoint，`epoch_0000.ckpt` 对应完成第1轮，`last.ckpt` 用于恢复。训练内验证记录当前/未来特征误差；检测 AP 使用下一节的独立评测命令。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m foresight.train_future_event \
  --config configs/event_full_joint.yaml --data-root /path/to/DSEC \
  --resume outputs/full_joint_seed42/checkpoints/last.ckpt \
  --output-dir outputs/full_joint_seed42
```

恢复时使用原 GPU 数。完整 checkpoint 恢复优化器、训练目标、种子及进度；epoch 中途恢复可能重读该轮开头的样本。

### 完整当前/未来评测

每次评测用一张 GPU。下面同时评估当前任务和未来任务，并分别接原检测头与真实 RGB 特征训练的检测头，输出四组 AP、APs、逐类指标、特征误差及样本预测。

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.diagnose_readout \
  --config configs/event_full_joint.yaml --data-root /path/to/DSEC \
  --encoder-checkpoint outputs/full_joint_seed42/checkpoints/last.ckpt \
  --readout-checkpoint checkpoints/readouts/rgb_current/roundtrip_last.pt \
  --full-split --split val --output outputs/full_joint_seed42_val
```

用相同命令评测 A 组及各 epoch checkpoint，在完整验证集上比较/选择模型，再将 `--split val` 改成 `--split test` 做完整测试。每次使用新的输出目录。去掉 `--encoder-checkpoint` 可得到相同评测协议下的 RGB 对照。

`--full-split` 使用完整划分；`--manifest configs/val600.json` 用于旧子集复测。第二个检测头的 `plan.json` 必须与权重放在同一目录且不作修改；其中记录的旧路径无需在新机器上存在。

## 开发

`main` 保持 RGB 基线，新实验使用独立分支：

```bash
python -m pytest -q
git checkout main
git checkout -b codex/experiment-name
```

权重和 `outputs/` 不提交到 Git。上游来源与许可证见 [SOURCES.json](foresight/vendor/SOURCES.json) 和 `foresight/vendor/`。

## Event Encoder（本分支）

E2 将 DINOv2 的输入投影改成 10 通道，微调整个 Encoder，用当前配对 RGB 的 PCA 特征做 SmoothL1 对齐。每个时刻使用曝光结束前 50ms 的事件，正负极性各 5 个等宽 bin；PCA、RGB Teacher、Predictor 和 Detector 均保持固定。推理只需要四段历史事件。

```bash
python -m pip install -e '.[event,test]'
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train --stage event
```

默认从 RGB 权重初始化，沿用固定 train6000/val600；两卡有效 batch 为 8。已有 E2 权重位于 `checkpoints/event/e2_7500.pt`，作为新训练初始化时加 `--init-from checkpoints/event/e2_7500.pt`。`--resume` 仅用于新项目产生的 Event 训练 checkpoint。

通过固定 Predictor/Detector 评估 Event-only 未来检测：

```bash
CUDA_VISIBLE_DEVICES=3 python -m foresight.evaluate --config configs/event.yaml \
  --encoder-checkpoint checkpoints/event/e2_7500.pt \
  --split val --output outputs/event_val.json
```

也可传入新训练的 `outputs/event_*/checkpoints/last.ckpt`。评估的事件分箱协议从 Encoder checkpoint 读取，避免配置与权重不匹配。

独立推理使用 `python -m foresight.infer_event --help`：输入含绝对微秒时间戳 `t`、原图坐标 `x/y` 和极性 `p` 的 NPZ，以及四个历史曝光结束时刻；不需要 RGB 或标签。旧实验权重可通过 `scripts/import_event_weights.py --source ... --output ...` 导出为新资产，优化器状态不迁入。

## 时间采样（本分支）

本分支基于 `codex/event-encoder`，只扩展每个历史时刻之前的事件窗口与分箱。五个桶按从早到晚排列，计数统一换算成等效 10ms 计数，再做 log/截断/resize；四个输入截止时刻与未来目标保持不变。

| 配置（`configs/`） | 窗口 | 桶宽，ms（从早到晚） |
|---|---:|---|
| `event_50ms_uniform.yaml` | 50ms | 10 / 10 / 10 / 10 / 10 |
| `event_50ms_recent_fine.yaml` | 50ms | 20 / 12 / 8 / 6 / 4 |
| `event_50ms_old_fine.yaml` | 50ms | 4 / 6 / 8 / 12 / 20 |
| `event_200ms_uniform.yaml` | 200ms | 40 / 40 / 40 / 40 / 40 |
| `event_200ms_recent_fine.yaml` | 200ms | 100 / 50 / 25 / 15 / 10 |

这些配置均从 E2-7500 初始化，重置优化器，训练 1500 次更新；两卡有效 batch 为 8。200ms 窗口之间存在重叠，并非四段互不相交的 200ms 事件。

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train --stage event \
  --config configs/event_200ms_recent_fine.yaml

CUDA_VISIBLE_DEVICES=3 python -m foresight.evaluate --config configs/event.yaml \
  --encoder-checkpoint checkpoints/event/200ms_recent_fine.pt \
  --split val --output outputs/event_200ms_val.json
```

`200ms_recent_fine.pt` 是迁入的已有权重。`configs/event_test600.json` 保留已使用过的 test 子集，可通过 `--split test --manifest configs/event_test600.json` 复测旧比较。
