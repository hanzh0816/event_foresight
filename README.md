# Event Foresight

独立的 **RGB DINO-Foresight → 未来目标检测** 基线。`main` 只维护可复现的 RGB 模型、DSEC 数据处理、训练和推理；后续实验从此分支建立。

```text
四张历史 RGB → DINOv2-B/14-reg → 固定 PCA → DINO-F Predictor
                                         → inverse PCA → Deformable DETR
```

编码器拼接第 2/5/8/11 层的 3072 维特征，PCA 压缩到 1152 维。输入沿用 448×896 预处理和四个相隔约 50ms 的历史时刻，预测 +50ms 的目标。输出框使用原图 640×480 像素坐标。

## 安装与权重

```bash
cd /home/hzh/code/event/event_foresight
python -m pip install -e '.[test]'
python scripts/link_checkpoints.py
```

已验证环境：Python 3.12、PyTorch 2.14.1、Lightning 2.6.6。运行需要 CUDA；CPU 用于数据、权重和小模型测试。DINOv2/DINO-Foresight 必要模型源码随项目提供，无需下载源码或使用 torch.hub。

`checkpoints` 是指向 `/data3/hzh/models/event_foresight/` 的软链接，包含：

| 文件 | 用途 |
|---|---|
| `dinof.pt` | 官方 RGB Encoder、PCA、Predictor 参数 |
| `detector.pt` | 已训练的 DSEC RGB Detector 参数 |
| `assets.json` | 来源哈希、导出哈希和张量一致性核验 |

这些文件只包含模型参数及必要元信息，没有旧优化器状态。权重、软链接和 `outputs/` 均由 Git 忽略。另一台机器可将相同资产放到自己的目录，然后运行：

```bash
python scripts/link_checkpoints.py --root /path/to/shared/checkpoints
```

需要从原始官方和 RGB 检测头 checkpoint 重新生成资产时，在项目根目录运行一次：

```bash
python -m scripts.import_weights \
  --dinof-source /path/to/dinof_highres.ckpt \
  --detector-source /path/to/rgb_detector.ckpt \
  --output-dir /path/to/shared/checkpoints
```

导入脚本逐项验证 404 个官方模型张量和 296 个检测头张量，保留 PCA 和标准化缓冲量。日常运行只读取导出文件，不依赖原项目及原 checkpoint 的配置路径。

## 推理与评估

四张原始大小图片按时间从早到晚传入，假定相邻图片间隔为 50ms。推理不需要标签、未来图片或 DSEC 数据目录。

```bash
CUDA_VISIBLE_DEVICES=3 python -m foresight.infer \
  --images frame0.png frame1.png frame2.png frame3.png \
  --output outputs/prediction.json
```

评估默认加载官方 Predictor 和已有 RGB Detector。数据根目录在 YAML 中指定，也可通过命令覆盖：

```bash
CUDA_VISIBLE_DEVICES=3 python -m foresight.evaluate \
  --split val --data-root /data3/hzh/DSEC \
  --output outputs/rgb_val.json
```

`--manifest selection.json` 可指定固定集合，接受 `{"val":{"sample_ids":["sequence:39", ...]}}`、`{"sample_ids":[...]}` 或字符串列表。样本顺序按清单保持。`--max-samples 8` 用于短验证。指标 JSON 使用 0–1 数值，并另存每个样本的预测结果。

`configs/val600.json` 保存迁移验收的固定集合。官方权重及现有 RGB 检测头在该集合的 AP/AP50 为 **39.9597/65.0568**，新旧实现的600条预测和全部指标完全一致；这不是完整 val 或 test 成绩。

评估默认 batch=1、Encoder chunk=1、BF16 和 `float32_matmul_precision=high`。比较历史结果时保持这些设置一致。单窗口推理和独立评估使用第一张可见 GPU。

## 分阶段训练

默认路线是直接使用官方 Predictor，仅训练 Detector。当前检测头参数用于初始化，优化器重新建立；如需从头训练检测头，将 `model.detector_checkpoint` 设为 `null`。

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train \
  --stage detector --config configs/detector.yaml
```

Predictor 的 DSEC 适配是独立、可选的阶段，不需要 Detector 权重。仅 Predictor 更新，冻结 RGB Encoder/PCA，真实未来 RGB 只用于构造监督目标。默认配置是 500 次优化器更新，不能将该预算视为充分训练。

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train \
  --stage predictor --config configs/predictor.yaml
```

两个阶段均自动使用全部可见 GPU，无需指定设备数量。`data.batch_size` 是每卡 batch；有效 batch 是每卡 batch × GPU 数 × `accumulation_steps`，实际数量会写入日志。训练产物默认位于 `outputs/<stage>_<time>/`，包含配置、CSV/TensorBoard 日志和完整 checkpoint。每次启动使用新的 `logs/version_N` 和 `tensorboard/version_N`，恢复训练保留先前曲线。

恢复新项目产生的同阶段 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m foresight.train --stage detector \
  --resume outputs/detector_RUN/checkpoints/last.ckpt
```

恢复包含模型、优化器、调度器和 Lightning 进度；优先在 epoch 边界恢复。epoch 中途恢复可能重读该 epoch 开头的样本，不承诺逐样本复现未保存的数据迭代位置。迁入的 `dinof.pt`/`detector.pt` 用于初始化，不能作为 `--resume` 文件。

评估新训练的检测头可加 `--checkpoint outputs/.../checkpoints/last.ckpt`。使用适配 Predictor 时，在检测头训练配置中填写 `model.predictor_checkpoint`，或在评估/推理时提供 `--predictor-checkpoint`；显式 Predictor 替换优先，检测头 checkpoint 不会悄悄把它覆盖回官方参数。加载时会检查冻结 Encoder/PCA 是否匹配。

短验证示例：

```bash
CUDA_VISIBLE_DEVICES=3 python -m foresight.train --stage detector \
  --max-steps 2 --batch-size 1 --num-workers 0 \
  --limit-train-batches 2 --limit-val-batches 1 \
  --output-dir outputs/smoke_detector
```

## 数据与项目约定

- 固定 DSEC 划分：41 个 train、6 个 val、12 个 test 序列，保留原八类顺序。
- 40 帧用于判断样本资格和时间连续性，模型实际只读取最后 4 帧。未来目标是下一帧；沿用有效非空未来标签筛选及原有框裁剪规则。
- RGB 数据读取仅需要 PNG、时间戳和 `tracks.npy`，不依赖事件文件或 HDF5。
- 配置内的相对资源路径以项目根目录解析，模型与数据处理不依赖启动时的工作目录。
- Predictor、Detector 训练的输入/目标分开；推理输入只包含历史 RGB。

```text
foresight/models/  模型与检测损失
foresight/data/    RGB 读取、预处理和样本窗口
foresight/vendor/  必要上游代码、许可证及来源版本
configs/          两个阶段的配置
scripts/          权重导出与软链接
tests/            数据、模型、训练和独立运行检查
```

```bash
python -m pytest -q
git checkout main
git checkout -b codex/experiment-name
```

新实验在独立分支修改模型、配置和训练方案，`main` 保持这份 RGB 基线。上游代码来源见 `foresight/vendor/SOURCES.json`，检测模块许可证见 `foresight/models/APACHE-LICENSE`。

## 迁移验收

2026-11-08：58 项测试通过（包含真实资产检查）。700 个导出张量与来源逐项一致；六个序列的 132 项中间特征和输出比较完全一致。固定 val600 的 AP/AP50 差值均为 0 个百分点。

两个训练阶段均完成单卡 2 步、保存、恢复至 3 步，以及双卡各 2 步检查；冻结模块保持不变，实际全局 batch 和验证样本数符合设置。已从 `/tmp` 禁止导入旧项目模块，验证四图推理、新训练权重加载和评估。只执行迁移验收，未启动正式训练。

本地详细证据保存在 `outputs/baseline_equivalence.json`、`outputs/forward_parity.json`、`outputs/migration_smoke/` 和 `outputs/cli_acceptance/`。复测真实权重加载：

```bash
FORESIGHT_ASSET_DIR=/data3/hzh/models/event_foresight python -m pytest -q
```
