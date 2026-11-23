# Event Foresight

基于 DINO-Foresight 的 RGB 未来目标检测基线：四张历史 RGB → DINOv2 → PCA → Predictor → inverse PCA → Detector。

输入为四张按时间排序、间隔约 50ms 的 640×480 图片，预处理至 448×896；预测未来 50ms 的目标，输出原图坐标下的检测框、类别和分数。

相关文献：[Event 特征提取、处理与对齐调研汇总](docs/event-representation-survey.md)，包含编码器、时序记忆、RGB/VFM 对齐、预测与生成、应用模块及各方法的 GitHub 入口。

## Event–RGB 动态预测研究

当前方案为 **Event motion-conditioned DINO-Foresight**：保留完整的 12 层 RGB Predictor，在第 6 层后由未来 token 读取历史事件的时空 tokens，注入条件残差，再预测未来 50ms 的场景特征。训练保留冻结的 DINOv2 与标准化 PCA，以真实未来 RGB 特征监督；先训练事件条件分支，再以较低学习率联合微调 Predictor。

从 [新版方法与训练交接](docs/research/DINOF_MOTION_CONDITIONING.md) 开始，使用 `configs/research/conditioning_*.yaml` 和 `python -m foresight.research.train/evaluate`。事件有序、事件时间汇聚、RGB 微调与冻结 RGB 共同使用因果截止、对齐视角和配对样本协议。代码从 `codex/event-wm-fusion` 的 `4132d60` 开始，当前分支为 `codex/event-dinof-motion-conditioning`。

[旧实验交接](docs/research/EXPERIMENT_HANDOFF.md) 与 [旧研究方案](docs/research/event-wm-plan.md) 保留此前融合、转移教师和运动 rollout 的历史设计；旧阶段不是运行当前方案的前置条件。新版尚无真实 DSEC 训练结果。

## 安装与权重

```bash
cd /home/hzh/code/event/event_foresight
python -m pip install -e '.[test]'
python scripts/link_checkpoints.py
```

运行需要 CUDA。默认权重目录为 `/data3/hzh/models/event_foresight/`，包含 `dinof.pt`、`detector.pt` 和 `assets.json`；`checkpoints` 通过软链接访问它们。使用其他目录时，先准备好这些文件，再建立链接：

```bash
python scripts/link_checkpoints.py --root /path/to/checkpoints
```

DSEC 数据根目录在 `configs/*.yaml` 的 `data.root` 中设置。配置内的相对路径以项目根目录解析，输出默认保存到 `outputs/`。

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

## 开发

`main` 保持 RGB 基线，新实验使用独立分支：

```bash
python -m pytest -q
git checkout main
git checkout -b codex/experiment-name
```

权重和 `outputs/` 不提交到 Git。上游来源与许可证见 [SOURCES.json](foresight/vendor/SOURCES.json) 和 [检测模块许可证](foresight/models/APACHE-LICENSE)。
