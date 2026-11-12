# Event 条件 Predictor 全量微调

本分支固定已选定的 `full_joint` Event Encoder，只更新 Predictor。训练输入是截止时刻之前的四个事件窗口，监督是真实下一帧 RGB 经冻结 Teacher/PCA 得到的特征：

```text
历史 events → 冻结 Event Encoder → 可训练 Predictor → 预测未来 PCA 特征
未来 RGB   → 冻结 RGB Teacher/PCA               → SmoothL1 目标
```

Detector 不在训练图中，没有框、类别或检测损失；当前对齐损失也不参与。现有 DSEC 窗口清单仍沿用项目的 GT 可用性筛选。Encoder 的 `full_joint` checkpoint 由用户选择，loader 会检查格式、固定坐标和事件编码，不能根据文件名证明它是哪次实验。

## 训练与恢复

默认 `configs/event_predictor_full.yaml` 使用完整 train/val 合格窗口，stride=1、无 manifest、无数量截断，10 epochs、无限 max_steps、每 GPU batch=1、梯度累积=4、常量 LR=1e-5。事件输入沿用 full_joint 的 50ms uniform。有效 batch 是 `GPU数 × batch_size × accumulation_steps`。

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m foresight.train_event_predictor \
  --config configs/event_predictor_full.yaml \
  --data-root /path/to/DSEC \
  --dinof-checkpoint /path/to/dinof.pt \
  --encoder-checkpoint /path/to/full_joint/checkpoints/last.ckpt \
  --output-dir outputs/event_predictor_full
```

初始 Predictor 使用配置的 RGB DINO-Foresight 模型；如原基线采用单独适配过的 RGB Predictor，可在配置中设置 `model.predictor_checkpoint`。旧 Encoder checkpoint 内自带的 Predictor 不覆盖这个明确选择。冻结 Encoder 的输入分箱必须与 checkpoint 一致，不能在此任务中顺便换成 200ms。

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m foresight.train_event_predictor \
  --config configs/event_predictor_full.yaml \
  --data-root /path/to/DSEC \
  --dinof-checkpoint /path/to/dinof.pt \
  --resume outputs/event_predictor_full/checkpoints/last.ckpt \
  --epochs 10 --output-dir outputs/event_predictor_full
```

恢复包含 Event Encoder、RGB Teacher、PCA、Predictor、optimizer、scheduler 和步数，不需要原 `full_joint` 文件仍在原路径。当前 `dinof.pt` 用于构造架构，完整参数随后由 resume checkpoint 恢复。恢复保留已保存的 batch、梯度累积、GPU 数、seed、precision、数据选择与事件编码；允许迁移 DSEC 路径和调整 worker 数。`--epochs`、`--max-steps` 是包含已完成训练的总预算。希望完整遍历后续 epoch 时优先从 `epoch_*.ckpt` 恢复；中途 step 截断恢复沿用 Lightning 的数据进度语义。

## 评价与推理

新 checkpoint 的 stage 是 `event_predictor`，同时保存配套冻结 Encoder 和训练后的 Predictor。最简单的评测方式是将它整体传给 `--encoder-checkpoint`。评测配置必须提供 Detector；训练配置的 `detector_checkpoint: null` 不能直接用于检测评价。

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m foresight.evaluate \
  --config configs/detector.yaml --data-root /path/to/DSEC \
  --encoder-checkpoint outputs/event_predictor_full/checkpoints/last.ckpt \
  --split val --output outputs/event_predictor_full/metrics_detection_val.json

CUDA_VISIBLE_DEVICES=0 python3 -m foresight.diagnose_readout \
  --config configs/detector.yaml --data-root /path/to/DSEC \
  --encoder-checkpoint outputs/event_predictor_full/checkpoints/last.ckpt \
  --full-split --split val --output outputs/event_predictor_full/readout_val
```

`configs/detector.yaml` 的 `dinof_checkpoint` 和 `detector_checkpoint` 应指向本机对应资产；可以复制为运行配置后修改这两个路径。已有 Detector 完整训练 checkpoint 可以通过 evaluate 的 `--checkpoint` 传入，导出资产 `detector.pt` 应配置在 YAML 中。

也可以保留原 Encoder 参数，并显式传入新 Predictor：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m foresight.evaluate \
  --config configs/detector.yaml --data-root /path/to/DSEC \
  --encoder-checkpoint /path/to/full_joint/checkpoints/last.ckpt \
  --predictor-checkpoint outputs/event_predictor_full/checkpoints/last.ckpt \
  --split val --output outputs/event_predictor_full/metrics_paired_val.json
```

加载顺序是 RGB/Detector 后端 → Event Encoder → 最终 Predictor。显式 `--predictor-checkpoint` 优先于 bundle 中的 Predictor；旧 RGB Predictor override 仍受支持。Event Predictor 必须与实际 Event Encoder **所有参数及 buffer**、事件编码、RGB Teacher、PCA 完全一致，不匹配会拒绝。它不能被 RGB-only Predictor loader 接受。旧 Event Encoder checkpoint 保持原有行为，只恢复 Encoder。

`diagnose_temporal` 和 `infer_event` 使用同一个装配入口，同样支持 bundle 及显式 Predictor override。原始事件推理示例：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m foresight.infer_event \
  --config configs/detector.yaml \
  --encoder-checkpoint outputs/event_predictor_full/checkpoints/last.ckpt \
  --events /path/to/history_events.npz \
  --frame-ends-us 1000000 1050000 1100000 1150000 \
  --output outputs/event_predictor_full/prediction.json
```

四个时间戳替换为实际历史窗口结束时刻；NPZ 不得包含最后截止时刻及之后的事件。推理不读取 RGB。评价及诊断输出会记录实际使用的 Predictor 来源，包括从 bundle 自动解析的路径。

## 记录与比较

运行输出包括配置与预算、train/val 清单数量及 hash、每 rank 样本访问顺序、首步梯度审计、累计样本数、每 epoch 完整 checkpoint 和最后 checkpoint。梯度审计要求 Predictor 有有限非零梯度，所有冻结模块无梯度。`metrics_val.json` 是未来特征 SmoothL1；下游检测 AP 由独立 evaluate/diagnose 计算。

同一 full_joint Encoder、检测头、验证清单下比较原 Predictor 与适配后的 Predictor，再看 AP/AP50、完整类别 AP 及数量。此实现尚未运行服务器全量训练，配置中的 10 epochs 是可调整实验预算。后续研究目标记录在 [future-explorations.md](future-explorations.md)。
