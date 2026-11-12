# 200ms recent_fine：全样本 Encoder joint 训练

分支：`codex/event-200ms-joint`。训练目标与 50ms 的 `full_joint` 相同，只扩大每个状态的事件窗口并改变分箱。只更新 Event Encoder；RGB Teacher、PCA、Predictor 均冻结，Detector 不参与训练。

## 输入与成对比较

| 项目 | 50ms 对照 | 本分支 |
| --- | --- | --- |
| 配置 | `configs/event_full_joint.yaml` | `configs/event_full_joint_200ms_recent_fine.yaml` |
| 每个状态的窗口 | 50ms | 200ms |
| 五个桶，从早到晚 | 10 / 10 / 10 / 10 / 10ms | 100 / 50 / 25 / 15 / 10ms |
| 历史状态数 / 输入通道数 | 4 / 10 | 4 / 10 |
| 四个状态的截止时刻 | 原曝光结束时刻，间隔约 50ms | 相同 |
| 名义预测时域 | 50ms | 50ms |
| 损失 | 四个历史状态对齐 + 真实未来 RGB 特征预测 | 相同 |

四个 200ms 窗口重叠，事件并集约覆盖 350ms 历史；没有把预测时域改为 200ms。每个桶的计数先乘 `10ms / 桶宽`，再应用已有 log、截断和 resize，因此沿用已验证的分箱与因果输入代码。

默认从同一 RGB 预训练权重初始化，不从旧 E2 或已训练的 50ms Encoder 继续。与 50ms 全量 joint 一样，不使用子集 manifest，训练 10 epochs，seed 42，学习率 `1e-4`，每卡 batch 1、累积 4 次；两卡有效 batch 为 8。当前／未来损失权重均为 1，`trainable_predictor: false`。比较时保持 GPU 数、训练预算和评测头一致。

## 数据检查与训练

在仓库根目录准备 `checkpoints/dinof.pt`。训练本身不需要 Detector 权重；评测另需 `checkpoints/detector.pt`。迁移方式见主 README。

```bash
python3 scripts/check_dsec.py --data-root /path/to/DSEC --event-window-us 200000

CUDA_VISIBLE_DEVICES=0,1 python3 -m foresight.train_future_event \
  --config configs/event_full_joint_200ms_recent_fine.yaml \
  --data-root /path/to/DSEC \
  --output-dir outputs/full_joint_200ms_recent_fine_seed42
```

首次运行可使用独立输出目录并追加 `--epochs 1 --limit-train-batches 8 --limit-val-batches 2`。短测与正式训练不要使用同一输出目录。CLI 的 `--dinof-checkpoint /path/to/dinof.pt` 可覆盖主干资产位置。

继续训练使用完整 checkpoint，恢复优化器、种子、损失和事件分箱协议：

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m foresight.train_future_event \
  --config configs/event_full_joint_200ms_recent_fine.yaml \
  --data-root /path/to/DSEC \
  --resume outputs/full_joint_200ms_recent_fine_seed42/checkpoints/last.ckpt \
  --epochs 10 --max-steps -1 \
  --output-dir outputs/full_joint_200ms_recent_fine_seed42
```

`--epochs` 表示包含已完成轮次的总轮数。恢复时保持原 GPU 数，优先使用 epoch 边界 checkpoint；更换配置文件不会把保存的 50ms 训练变成 200ms，恢复以 checkpoint 协议为准。`--init-from` 是权重初始化，属于另一种续训实验，不应与本分支默认的从 RGB 初始化直接混作成对对照。

## 未来检测评测

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m foresight.evaluate \
  --config configs/event_full_joint_200ms_recent_fine.yaml \
  --data-root /path/to/DSEC --split val \
  --encoder-checkpoint outputs/full_joint_200ms_recent_fine_seed42/checkpoints/last.ckpt \
  --output outputs/full_joint_200ms_recent_fine_seed42_val.json
```

不传 `--manifest` 或 `--max-samples` 时评测完整划分；事件窗口与分箱从 Encoder checkpoint 读取。指定其他检测头时使用 `--checkpoint /path/to/detector.ckpt`。使用同一检测头复评 50ms 对照，报告全部类别 AP 与 GT 数、car/pedestrian、AP50/AP75 和小目标 AP。

当前／未来双头诊断沿用主 README 的 `foresight.diagnose_readout`，换成本分支配置和 Encoder checkpoint，并使用 `--full-split`。

## 产物与状态

保留每轮 `epoch_*.ckpt`、最终 `last.ckpt`、解析后的配置、数据清单统计、梯度审计和训练进度。检测 AP 通过独立评测获得，不以特征误差代替任务成绩。

本分支提供训练配置与验证，不代表已完成真实 DSEC 全量训练。监督分配、多时域预测和原生事件动态目标记录在 [后续探索](future-explorations.md)，本分支未实现。
