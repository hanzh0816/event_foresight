# Event motion-conditioned DINO-Foresight：方法与训练交接

更新日期：2026-11-23。实现分支：`codex/event-dinof-motion-conditioning`，基点：`codex/event-wm-fusion@4132d60`。本文是当前方案入口；[旧方案](event-wm-plan.md) 和 [旧三阶段交接](EXPERIMENT_HANDOFF.md) 仅供追溯。

## 1. 研究问题与范围

在相同历史 RGB 与观测截止时刻下，事件记录的帧间变化过程能否改善预训练场景特征的未来预测？进一步，有序的事件子窗是否比同一事件历史的时间汇聚更有效？

RGB 预训练负责场景语义和空间上下文，完整 DINO-Foresight Predictor 负责场景演化。事件分支保留历史区间内的时间组织，由 Predictor 的未来 token 按预测需要读取，再影响后续场景预测。这里的 motion condition 是由历史事件构造、受未来场景目标训练的隐变量；其物理运动含义需要实验验证。

第一轮只做无动作条件的未来单步检测：输入 4 张历史 RGB 和其间 3 个 Event 区间，输出下一真实 RGB 时刻的场景 latent，经共同冻结检测头得到未来目标框。名义间隔为 50ms，实际时间差记录在评价结果中。

方法重点是事件过程如何进入已有场景预测器，以及这种时间信息是否贡献未来任务收益。仅加入 cross-attention、训练两个阶段或联合微调本身不构成已证实的创新性。当前无真实 DSEC 训练结果、未来检测收益或泛化结论。

## 2. 模型与数据流

### RGB 场景空间

沿用导出的 `dinof.pt`：冻结 DINOv2 ViT-B/14，拼接第 2、5、8、11 层特征，得到每个 patch 的 3072 维表示。输入 RGB 从 640×480 预处理至 448×896，patch 网格为 32×64，共 2048 个 tokens。

保留 checkpoint 中固定的标准化统计量与 PCA：

```text
x = frozen_DINO(I)                          # [B, N=2048, 3072]
z = (((x - mean) / std) - pca_mean) @ W.T   # [B, N, 1152]
x_hat = (z_hat @ W + pca_mean) * std + mean
```

`mean/std/pca_mean/W` 全部冻结，沿用原资产；不在 DSEC 上重估，不额外 whitening。预测器和检测头之间继续使用这一固定坐标空间。RGB Predictor 保留完整 12 层、1152 维 hidden 和原预训练权重，约 257.76M 参数；默认新增条件分支为 505,248 参数。`event_ordered` 与 `event_pooled` 参数量一致，`rgb` 不增加分支参数。

### 历史事件表示

设四个历史 RGB 的曝光结束时间为 `s0 < s1 < s2 < s3`。读取 `[s0,s1)`、`[s1,s2)`、`[s2,s3)` 三个相邻事件区间，每段等分为 5 个时间 bins，分别累计正负极性事件并取 `log1p`。当前事件体素输入分辨率为 128×256。

每个 bin 独立经过共享 CNN，得到与 RGB patch 网格对应的空间 tokens。编码器保留 `3 intervals × 5 bins` 的时间轴，提供区间/bin 的时间位置信息。它从当前预测目标初始化训练，不加载旧 Event-only RGB 对齐编码器。

`event_ordered` 保留 bin 顺序。`event_pooled` 在每个区间内部对 5 个 bin 特征取均值并复制，再加入与有序组相同的区间/bin 位置编码。它保留 3 个历史区间的先后关系，去掉每个区间内部的 bin 顺序；两组使用相同事件范围、空间处理、参数量与 attention 形状。

### 第 6 层后的条件注入

Predictor 读取四帧场景 latent 和可学习的未来 mask token，正常运行前 6 个时空 Transformer blocks。第 6 个完整 block 之后，未来 token 的 hidden state 作为 query，读取对应空间位置的事件时间 tokens：

```text
h = DinoF.blocks[0:6](RGB history tokens + future mask)
q_p = query_projection(h_future,p)
c_p = temporal_cross_attention(q_p, event_tokens_at_patch_p)
h_future,p = h_future,p + zero_initialized_projection(c_p)
z_hat = DinoF.blocks[6:12](h).future_prediction
```

每个位置的查询只沿事件时间轴读取，CNN 提供邻域空间感受野；后半段 DinoF 的空间注意力继续传播场景信息。注入仅更新未来 token，历史 RGB token 沿用原主干计算。条件输出投影零初始化，使初始预测保持预训练 RGB 路径，随后由场景损失学习条件残差。

条件分支输出是 Predictor hidden 中的残差；事件 tokens 不需要被蒸馏成 RGB PCA latent。方案使用单一未来场景目标，不引入物理 flow 标签、转移教师、未来事件 latent 损失或独立未来运动 rollout。

### 监督与梯度

```text
history RGB ── frozen DINO/PCA ── full DinoF with event condition ── z_hat
history Event ── trainable per-bin CNN + condition reader ──────────┘
future RGB ── frozen DINO/PCA ── z_target ── scene loss(z_hat, z_target)
```

真实未来 RGB 仅构造训练目标与可选特征评价，不进入预测输入。conditioning 数据路径不加载未来 Event 目标。训练检测头保持冻结，检测标签用于评价；当前配置使用 `scene_weight: 1.0`、`motion_weight: 0.0` 与 `smooth_l1`，实现采用 PyTorch 默认 `beta=1`。

冻结 Predictor 权重的 warmup 仍需保留其后半段计算图：梯度要经过第 7–12 层回到条件分支。只有 RGB 编码器与目标编码可以在 `no_grad` 下执行。

## 3. 配置与训练阶段

统一入口为 `python -m foresight.research.train` 和 `python -m foresight.research.evaluate`，所有新版配置使用 `stage: conditioning`。

| 配置 | 模式与可训练部分 | 用途 |
|---|---|---|
| `conditioning_warmup.yaml` | `event_ordered`；仅事件编码与条件读取/注入分支 | 从原 RGB Predictor 起步，学习条件残差 |
| `conditioning_joint.yaml` | `event_ordered`；条件分支与完整 Predictor | 从 warmup 权重初始化，以较低 Predictor LR 联合训练 |
| `conditioning_rgb.yaml` | `rgb`；完整 Predictor | 相同未来场景监督与共同预算的 RGB 微调对照 |
| `conditioning_pooled.yaml` | `event_pooled`；仅条件分支 | 与 ordered warmup 匹配 |
| `conditioning_pooled_joint.yaml` | `event_pooled`；条件分支与完整 Predictor | 从 pooled warmup 续训，匹配 ordered joint |
| `conditioning_rgb_frozen.yaml` | `rgb`；所有模块冻结 | 原 RGB Predictor 的配对评价基线 |

冻结 DINO/PCA 与共同检测头贯穿所有阶段。`optimization.lr` 用于条件分支，默认 `1e-4`；`optimization.predictor_lr` 用于可训练 Predictor，默认 `1e-5`。默认 warmup 1,000 步、joint 4,000 步、RGB 微调 5,000 步均为初始实验预算，不是已确定的收敛配置。所有配置默认关闭 Predictor dropout 并启用 activation checkpointing；是否训练 Predictor 由 `model.train_predictor` 显式控制。

`model.predictor_checkpoint` 可选加载原项目的 RGB Predictor/Detector 训练 checkpoint，核验冻结 Encoder/PCA 一致后，仅替换共同的初始 RGB Predictor；默认 `null`，直接使用 `dinof.pt`。如果启用，所有对照必须使用同一来源，并单独记录该训练 checkpoint 的哈希。它与跨阶段 `model.init_checkpoint` 不能同时设置。

跨阶段续训与断点恢复分开使用：

- `model.init_checkpoint`：从前一 conditioning checkpoint 载入模型权重，开始新的优化器状态与预算，用于 warmup → joint。
- `--resume`：恢复当前配置对应的完整训练，包括优化器与步数。科学配置和优化器设置必须与存档相符，不用于切换训练阶段。

对照预算同时记录两项：总样本暴露/训练步数，以及 Predictor 实际更新步数。warmup 加 joint 的总预算与 RGB 微调不能仅按最后一次运行步数比较。为区分时间顺序影响，pooled 应使用与 ordered 相同的 warmup → joint 日程；从对应 pooled warmup 权重初始化 pooled joint。配置文件名不替代训练谱系与预算说明。

## 4. DSEC 因果、坐标和配对协议

### 必填数据与资产

- `data.root`：DSEC 检测格式根目录。
- `model.dinof_checkpoint`：官方 RGB Predictor 的项目 export，默认 `checkpoints/dinof.pt`。
- `model.detector_checkpoint`：共同冻结 future detector export，默认 `checkpoints/detector.pt`。
- `data.alignment_evidence`：RGB/Event/检测框处于同一视角的实际预处理版本、标定和路径。

`data.alignment: prealigned_event_view` 是数据前提，不执行配准。必须使用已对齐的 RGB、Event 和框；单纯 resize 不能将不同相机视角变为共视角。保存配准来源和叠加可视化作为运行证据。

YAML 的 `extends` 相对当前 YAML 所在目录解析，数据与模型资产的相对路径基于仓库根目录；CLI `--output-dir` 相对执行命令时的工作目录解析，以下命令均在仓库根目录执行。建议为实际机器建立本地继承配置，保留已归档运行配置。

### 时间与输入范围

- 输入只有 4 帧 RGB 与连接它们的 3 个事件区间。截止时刻为最后一帧的曝光结束；HDF5 的 `t_offset` 由 `EventStream` 处理。
- 事件区间采用左闭右开 `[start,end)`；最后一段不包含截止时刻及之后的事件。
- `rollout_steps: 1`、`evaluation_horizons: [1]` 只评价下一真实帧。记录目标标签时间、曝光结束时间及其相对截止时刻的实际差值。
- `sample_history_frames: 40` 与原 RGB 基线的候选窗口筛选对齐，它不将输入扩大到 40 帧，也不将事件历史扩大到 40 帧。
- 采用项目固定 DSEC-det 序列划分与相同 stride，训练/验证/测试不随机拆分重叠窗口。

研究数据集还检查 Event 覆盖、曝光记录、目标和时间连续性；因此 `sample_history_frames: 40` 本身不保证与历史 RGB 报告的每个样本一致。正式比较统一使用此次研究的配对 inventory：相同 `sample_ids`、目标时刻、GT 集合与事件可用性筛选。RGB frozen、RGB 微调与两个 Event 组全部在该 inventory 重评，记录 digest 与样本/GT 数量。

评价可用同一 `--manifest` 固定所有模型的样本。`--max-samples` 只用于小批诊断，不能作为 full val/test 结果。AP 的现有约定排除没有有效 GT 的目标帧；所有对照必须保持相同规则。

## 5. 运行顺序与命令

### 环境与预检查

在拥有 DSEC、checkpoint 和 GPU 的训练机仓库根目录运行：

```bash
python -m pip install -e '.[test,event]'
python scripts/link_checkpoints.py --root /path/to/checkpoints
python scripts/check_dsec.py --data-root /path/to/DSEC
```

先为下列配置填入真实 `data.root`、资产路径和 `data.alignment_evidence`；以下命令使用仓库配置名，实际机器若采用继承配置，应替换为对应本地路径。所有新运行使用空输出目录，评价使用新的 JSON 文件名。

### Smoke 与 warmup

将以下内容保存为仓库根目录的 `conditioning_smoke.local.yaml`，继承已填写数据与资产信息的 warmup 配置：

```yaml
extends: configs/research/conditioning_warmup.yaml
optimization:
  epochs: 1
  max_steps: 2
  accumulation_steps: 1
runtime:
  devices: 1
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.research.train \
  --config conditioning_smoke.local.yaml \
  --output-dir outputs/research/conditioning-smoke \
  --limit-train-batches 2 --limit-val-batches 2

CUDA_VISIBLE_DEVICES=0 python -m foresight.research.train \
  --config configs/research/conditioning_warmup.yaml \
  --output-dir outputs/research/conditioning-warmup-seed42
```

该 smoke 最多进行两个 optimizer steps；`--limit-train-batches 2` 本身只限制每个 epoch 的 batch 数，必须配合上述 epoch、步数与梯度累积设置。Smoke 只验证所用小批的数据、梯度、存档和入口；正式 warmup 从独立空目录开始。它不代表完整数据集或收敛性已验证。

### 联合训练与恢复

建立 joint 的本地继承配置，例如将以下内容保存为仓库根目录的 `conditioning_joint.local.yaml`：

```yaml
extends: configs/research/conditioning_joint.yaml
model:
  init_checkpoint: outputs/research/conditioning-warmup-seed42/checkpoints/last.ckpt
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.research.train \
  --config conditioning_joint.local.yaml \
  --output-dir outputs/research/conditioning-joint-seed42

CUDA_VISIBLE_DEVICES=0 python -m foresight.research.train \
  --config conditioning_joint.local.yaml \
  --output-dir outputs/research/conditioning-joint-seed42 \
  --resume outputs/research/conditioning-joint-seed42/checkpoints/last.ckpt
```

运行前确认继承得到的数据与资产配置也已填写。joint 的配置保存原 warmup 来源；同阶段恢复使用 joint 的 `last.ckpt`。保持 batch/GPU/精度等设置可复现；epoch 中途恢复仍可能重读部分样本。

### 对照与评价

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.research.train \
  --config configs/research/conditioning_rgb.yaml \
  --output-dir outputs/research/conditioning-rgb-seed42

CUDA_VISIBLE_DEVICES=0 python -m foresight.research.evaluate \
  --config configs/research/conditioning_rgb_frozen.yaml \
  --split val --feature-metrics \
  --output outputs/research/conditioning-rgb-frozen-val.json

CUDA_VISIBLE_DEVICES=0 python -m foresight.research.evaluate \
  --config conditioning_joint.local.yaml \
  --checkpoint outputs/research/conditioning-joint-seed42/checkpoints/last.ckpt \
  --split val --feature-metrics \
  --output outputs/research/conditioning-ordered-val.json

CUDA_VISIBLE_DEVICES=0 python -m foresight.research.evaluate \
  --config configs/research/conditioning_rgb.yaml \
  --checkpoint outputs/research/conditioning-rgb-seed42/checkpoints/last.ckpt \
  --split val --feature-metrics \
  --output outputs/research/conditioning-rgb-tuned-val.json

python -m foresight.research.compare \
  outputs/research/conditioning-rgb-tuned-val.json \
  outputs/research/conditioning-ordered-val.json \
  --output outputs/research/conditioning-rgb-vs-ordered.json
```

同一 manifest 可追加到所有 `evaluate` 命令。pooled warmup 使用 `conditioning_pooled.yaml`，joint 使用 `conditioning_pooled_joint.yaml`，在本地继承配置中将 `model.init_checkpoint` 指向实际 pooled warmup 的 `last.ckpt`；其余训练/评价命令沿用有序组并使用独立输出目录。正式对照共同确定预算和 seeds，在 val 完成选择后再运行完整 test。

原始 RGB baseline 使用独立 `conditioning_rgb_frozen.yaml` 评价。联合训练后，在同一任务 checkpoint 上使用 `--reference rgb` 会关闭事件条件、评价当前已微调的 RGB backbone；它是移除条件的消融，报告会标记这一来源，不能作为原始冻结 RGB baseline。比较工具要求相同冻结 Encoder/PCA 空间和 detector，同时分别记录各自 Predictor 的状态与初始化来源；微调后 Predictor 权重不同是实验变量。

## 6. 必须回答的实验问题

| 对比 | 可支持的判断 |
|---|---|
| RGB 微调 vs RGB frozen | 目标数据上的 Predictor 训练本身带来多少收益 |
| Event ordered vs 同预算 RGB 微调 | 历史事件条件是否在成熟场景预测器之上提供额外信息 |
| Event ordered vs 同日程 Event pooled | 所用事件时间组织是否有效 |
| warmup vs joint | Predictor 适配是否有助于使用条件信息；同时报告额外预算 |

报告完整 val/test AP、AP50、AP75、类别/尺度指标、样本与 GT 数，并保留全图/前景/背景 latent 误差作为辅助诊断。AP 数值为 0–1，组间 AP 差用百分点表示。共同冻结 detector 曾适配 RGB Predictor 输出，因此真实未来 RGB latent 的读出仅是诊断参考，不能当作数学上界。

如果在合理共同预算、配对完整样本和重复 seeds 下，ordered 不能优于 RGB 微调，则当前事件条件方案缺乏任务收益依据；如果 ordered 与 pooled 相当，则不能声称细粒度时间顺序是收益来源。局部类别或少数序列改善不能替代完整集结果。

## 7. 交接产物与验证状态

训练与评价分别归档配置展开结果、Git revision、初始化 checkpoint 路径与哈希、冻结/可训练参数、batch/GPU/精度、预算、seed、数据配准来源，以及评价样本 inventory digest、真实时间差和 detector/feature-space/Predictor 哈希。全局 batch 按每卡 batch × GPU 数 × accumulation steps 计算。代码不自动归档未提交修改；运行前需额外保存 `git diff`，并保存未跟踪的本地配置文件。`last.ckpt` 是最后状态，不等于验证集最优模型。

主要实现入口：

- `foresight/models/forecaster.py`：冻结场景空间、完整 DinoF、条件注入接口。
- `foresight/research/conditioning.py`：事件逐 bin 编码与未来 token 条件读取。
- `foresight/research/task.py`：冻结策略、场景损失、参数组、初始化与 checkpoint。
- `foresight/research/data.py`、`foresight/data/event_reader.py`：因果输入、目标隔离和配对窗口。
- `foresight/research/train.py`、`evaluate.py`、`compare.py`：运行存档、未来检测与公平比较。

2026-11-23 开发验证：分组执行的局部测试共 75 个通过、1 个跳过（需要外部官方权重）。覆盖实际 DinoF blocks 的微型 CPU 前向/梯度、FP32/BF16 零初始化回退、条件注入位置、checkpointing、未来输入隔离、两个参数组、权重续训、完整优化器恢复及成对评价协议；这些测试使用合成数据，不是 DSEC 实验。后续补充的初始化谱系字段仅完成静态检查。

用户要求本服务器停止训练后，不再运行训练类测试或探针，仅进行静态代码与文档检查。本机没有 DSEC、生产 checkpoint 或 GPU；未执行真实 DSEC 训练、GPU/DDP 验证、完整集 AP 或收敛性实验。上文命令供具备环境的训练机使用。
