# Event–RGB 世界模型实验交接

本文是实验 Agent 的入口。先阅读 [研究方案](event-wm-plan.md)，再按本文件执行。代码实现的是研究假设的实验框架；没有预填真实 DSEC 训练结果。

## 1. 研究上下文与要回答的问题

目标是在视觉预训练场景表示的基础上，利用事件区间内的变化过程形成可预测、能驱动场景转移的运动表示，服务驾驶未来感知。模型输入是 RGB 与事件历史，输出为固定未来时刻的场景 latent，经共同检测头评价未来车辆、行人检测。研究范围为无动作条件预测。

三项核心命题分别是：

1. 在相同 RGB 历史与时间截止下，事件历史提供额外预测信息，区间内时间结构具有作用。
2. 事件编码出的紧凑运动表示能够解释场景转移。
3. 从历史预测运动表示并推进场景，比直接融合或仅增加辅助运动损失更有效。

预训练 RGB 模型提供场景知识；事件编码器从新目标独立初始化，不加载旧的 RGB 对齐 Event Encoder。运动 tokens 不被投影进 RGB PCA 空间。旧 Event-only 实验不作为本方案的性能依据。

## 2. 分支与依赖

三个分支累积实现，共用数据、训练和评价工具：

| 分支 | 内容 | 正式训练的前提 |
|---|---|---|
| `codex/event-wm-fusion` | 新事件编码、RGB/融合对照、共同数据与评价协议 | 数据就绪、坐标和时间协议确认 |
| `codex/event-wm-transition` | 在前一分支上增加已观测转移教师 | 检验事件动态能否解释视觉状态变化 |
| `codex/event-wm-motion-rollout` | 在前两分支上增加运动预测、辅助损失对照、联合转移与多步推演 | 获得有效的转移教师 checkpoint |

最新分支包含所有实验。无需为 A/B/C/D 分别维护代码。使用配置、seed 和运行 ID 区分实验，保留 `git rev-parse HEAD` 与工作区差异。不能将每个分支已有代码理解为所有阶段都应同时开展昂贵训练。

## 3. 模型与信息流

设 RGB 特征 `z_i = frozen_DINO_PCA(I_i)`；事件表示 `m_i = G(E_[i-1,i))`，其中 i 是对应观测端点，实际区间边界使用微秒时间。事件编码先按子窗分别提取空间特征，再混合时间维，输出空间排列的 motion tokens。

- Fusion：冻结 RGB 预测器得到 baseline future latent，训练小型条件残差模块修正预测。比较 RGB 条件、事件时间汇聚、事件有序表示。
- Transition：`G(真实区间 Event) -> m`；`T(起点 z,m) -> 终点 z`。RGB 终点仅用于监督。该路径使用了目标区间观测，是转移诊断，不是未来预测。
- Motion：冻结训练好的运动教师，`P_m(历史 z,m) -> 未来 m_hat`；`T(z,m_hat) -> 未来 z_hat`。未来事件仅生成固定监督目标。场景损失通过预测运动路径反传。
- Direct/Auxiliary：共用与 Explicit 完全相同的预训练转移器 T 和起点场景；Direct/Auxiliary 用独立的自由条件代码驱动 T，Auxiliary 额外监督另一 motion head。Explicit 则让受到运动监督的同一代码驱动 T。各组实例化相同两个预测头；Direct/Auxiliary 的场景 rollout 不使用 motion head 输出。这组实验检验监督与转移条件是否需要绑定；第一轮普通融合仍作为独立强对照。
- Joint：目标教师保持冻结；适配用于预测场景的转移器副本。多步训练使用模型自己的预测状态推进。

冻结转移器的参数不意味着禁用其输入梯度。任务代码需维持 `scene loss -> transition -> motion predictor` 路径。

## 4. 数据与评价协议

### 数据准备

安装依赖：

```bash
python3 -m pip install -e '.[test,event]'
python3 -m pytest -q
python3 scripts/check_dsec.py --data-root /path/to/DSEC
```

配置在 `configs/research/`。`extends` 相对配置文件位置解析；数据、权重、输出路径相对仓库根目录解析。建议复制配置至本地位置或创建继承配置；不要修改已归档运行的配置。

必须填写：

- `data.root`：DSEC 检测格式根目录。
- `model.dinof_checkpoint`：已有 RGB DINO-Foresight export。
- `model.detector_checkpoint`：共同的 RGB future detector export，用于评价。
- `data.alignment_evidence`：实际 RGB/Event/检测框共视角预处理或校准的来源、版本和路径。

`data.alignment: prealigned_event_view` 表示输入已经位于一致的事件视角；它不是执行配准的开关。代码进行体素化和缩放，不虚构跨相机几何映射。不能把普通 rectified RGB 与 raw Event 简单缩放后当成对齐数据。核查 RGB 重映射过程、事件坐标与框坐标，并保存叠加可视化证据。

### 时间边界与标签

- 四个历史 RGB 形成三个相邻观测区间，事件覆盖相同起点至最后 RGB 的曝光结束时刻。传感器 `t_offset` 由 EventStream 处理。
- 每个区间使用左闭右开的事件范围；记录曝光结束与真实目标时间戳，所以实际预测时长可能不恰好为 50ms。
- `history_rgb/history_events` 是预测输入；`future_rgb_target/future_events_target` 是独立目标。正式 forecast 入口不得读取目标字段。
- `rollout_steps: 4` 对应后四个真实帧标签，评价可选择第 1、2、4 步。所有对照使用同一份 inventory/manifest。
- 多步 inventory 使用统一可用目标集合，不允许各 horizon 任意过滤出不同样本后直接比较。
- 数据划分沿用固定官方序列名单；不得随机拆分高度重叠窗口为 train/val。

### 结果口径

评价输出 AP/AP50/AP75 和逐类别指标，单位为 0–1；比较工具导出百分点增量。`--feature-metrics` 同时记录全图、框覆盖前景及背景的 latent L1。没有相应区域时输出 null。

所有模型首先使用同一冻结 detector。该 detector 曾在 RGB predictor 的输出上训练，因此真实未来 RGB 特征的读出只是诊断参考，不是数学意义的上界。显著结论应再用同协议读出适配或独立任务检验，而不能把读出分布差异全部归因于世界模型。

现有评估沿用原项目 COCO 规则：无有效 GT 的目标帧不计入 AP；有效 GT 帧的漏预测仍被计入。必须报告实际样本、目标、类别数量；不要对只有很少 GT 的类别过度解释。

## 5. 第一轮：历史信息与时间结构

| ID | 配置 | 目的 |
|---|---|---|
| A | `fusion_rgb.yaml` | 原 RGB predictor，直接评价，无训练 |
| B | `fusion_rgb_control.yaml` | RGB 条件残差，控制新增可训练模块 |
| C | `fusion_pooled.yaml` | 事件空间证据，汇聚子窗时间 |
| D | `fusion_ordered.yaml` | 事件空间与子窗时间结构 |

C/D 使用相同事件范围、参数与计算形状。C 在共享空间编码之后，将子窗特征时间均值并复制，再走相同时间混合模块；它保留无序子窗分布，消除顺序，并非把原始事件累积成单张图。B 与事件 CNN 参数量不天然完全相等，必须报告可训练参数和训练资源；不得仅凭 B 对照宣称严格参数匹配。

先跑小批验证入口、梯度、checkpoint 与评价：

```bash
python3 -m foresight.research.train --config configs/research/fusion_ordered.yaml \
  --output-dir outputs/research/D-smoke --limit-train-batches 2 --limit-val-batches 2
```

正式 B/C/D 使用相同数据、预算和 seeds（建议 42、43、44），独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m foresight.research.train \
  --config configs/research/fusion_ordered.yaml --output-dir outputs/research/D-seed42
python3 -m foresight.research.evaluate --config configs/research/fusion_rgb.yaml \
  --split val --feature-metrics --output outputs/research/A-val.json
python3 -m foresight.research.evaluate --config configs/research/fusion_ordered.yaml \
  --checkpoint outputs/research/D-seed42/checkpoints/last.ckpt \
  --split val --feature-metrics --output outputs/research/D-val.json
python3 -m foresight.research.compare outputs/research/A-val.json outputs/research/D-val.json \
  --output outputs/research/AD-comparison.json
```

配置中的 2,000 steps 是启动预算，不是已确认的收敛预算。用训练/验证曲线确定正式共同预算，再运行完整对照。选型只用 val；确定设置后再运行 test。

判读：D>B 支持事件历史的有效性；D>C 支持所用时间结构的有效性。关注前景、定位敏感的 AP75、类别数量和不同序列，而非只看总 AP。若无收益，先检查数据与表示；允许小规模转移诊断定位瓶颈，不直接扩大完整模型。

## 6. 第二轮：观测转移教师

在 `codex/event-wm-transition` 或最新分支运行：

```bash
python3 -m foresight.research.train --config configs/research/transition.yaml \
  --output-dir outputs/research/teacher-ordered
python3 -m foresight.research.evaluate --config configs/research/transition.yaml \
  --checkpoint outputs/research/teacher-ordered/checkpoints/last.ckpt \
  --feature-metrics --output outputs/research/teacher-diagnostic.json
```

另训 `transition_pooled.yaml`、`transition_no_motion.yaml`。冻结 RGB encoder/PCA，训练 G/T。比较区间时间结构、是否使用运动条件，以及前景转移误差。诊断报告标记 `observed_transition_diagnostic`；比较工具拒绝将它与正式 forecasting 排名。

继续条件：运动条件在验证序列上有稳定作用，且模型没有仅复制起点/背景。记录 motion 样本方差、转移对 motion 的敏感性和匹配无运动对照。通过后固定教师 checkpoint 与 SHA256，供所有第三轮方法共同使用。

上述 `--feature-metrics` 命令同时写出 `transition_diagnostics`：运动变化统计、真实运动条件和零运动条件的转移误差、复制起点误差及输出差异。零运动干预仅用于检查敏感性；正式无运动对照仍需独立训练 `transition_no_motion.yaml`。多步时该诊断逐区间使用真实起点，主 AP 则使用完整 observed-event rollout，两者分别解释。

第二轮各组使用相同样本后，可用 `python3 -m foresight.research.compare <no-motion.json> <ordered.json> <pooled.json> --diagnostic --output <comparison.json>` 汇总诊断结果；该模式同样拒绝混入正式未来预测或真实未来 RGB 读出。

## 7. 第三轮：运动可预测性与结构价值

把 `motion.yaml` 的 `model.teacher_checkpoint` 指向已通过诊断的教师训练 checkpoint。未来 motion 监督始终来自冻结教师；模型输入只有历史。

| 配置 | 预测路径／作用 |
|---|---|
| `motion.yaml` | Event＋RGB 历史预测 motion，再经 T 预测场景 |
| `motion_rgb.yaml` | 仅 RGB 历史预测同一 motion 目标，分离推理时事件价值 |
| `motion_persistence.yaml` | 保持最后运动，通过相同 T 推演，仅评价、不训练；无需第三轮 checkpoint |
| `motion_direct.yaml` | 直接场景预测，无 motion loss |
| `motion_auxiliary.yaml` | 同样直接路径，加 motion loss |

```bash
python3 -m foresight.research.train --config configs/research/motion.yaml \
  --output-dir outputs/research/motion-explicit
python3 -m foresight.research.evaluate --config configs/research/motion.yaml \
  --checkpoint outputs/research/motion-explicit/checkpoints/last.ckpt \
  --feature-metrics --output outputs/research/motion-explicit-val.json
```

运动延续对照直接执行 `evaluate --config configs/research/motion_persistence.yaml --output ...`，省略 `--checkpoint`；它从指定的第二轮教师恢复 G/T。

Direct 输出的独立 motion head 没有运动监督，不用其 motion 误差评判场景预测能力。Auxiliary 的运动头虽然受监督，也不驱动未来场景；各组都必须比较场景预测结果。

训练检查：未来事件只进入固定目标分支；梯度到达 motion predictor；固定教师无梯度；场景损失确实经过预测 motion。不能仅报告 motion loss；同时评价共同读出的 future detection 与前景误差。

判读：Explicit 比 motion 延续强，支持学习动态演化的价值；Event 输入比 RGB-only 同目标强，支持历史事件的贡献；Explicit 比 Direct/Auxiliary 强，支持把运动监督绑定到实际驱动场景转移的代码。Direct/Auxiliary 也包含中间转移代码，不能由此宣称显式中间变量本身是唯一原因。

## 8. 多步与联合适配

使用 `motion_rollout.yaml` 训练固定转移器的四步预测；`motion_joint.yaml` 联合适配转移副本，监督教师不变。Direct/Auxiliary/RGB-input 对照均继承相同 `rollout_steps: 4`、样本集合与预算。不能把一步训练和四步训练的结果当成只改变结构的对照。

`model.init_checkpoint` 用于阶段间权重初始化：恢复同一教师、同一预测模式的场景与运动模型权重，重新创建优化器，记录来源 SHA256；允许改变 rollout 长度以及启用转移副本适配。`motion_joint.yaml` 默认从 `outputs/research/motion-rollout/checkpoints/last.ckpt` 初始化，运行前指向实际选定的 checkpoint。研究联合适配本身的作用时，同时从该 checkpoint 初始化固定 T 的对照，给两组相同的额外训练预算。Direct/Auxiliary/Explicit 的首次比较均从同一个第二轮教师开始，不能互相继承训练权重。

```bash
python3 -m foresight.research.train --config configs/research/motion_rollout.yaml \
  --output-dir outputs/research/motion-rollout
python3 -m foresight.research.evaluate --config configs/research/motion_rollout.yaml \
  --checkpoint outputs/research/motion-rollout/checkpoints/last.ckpt \
  --horizons 1 2 4 --feature-metrics --output outputs/research/rollout-val.json
# 相同 inventory 上递归运行冻结 RGB predictor：
python3 -m foresight.research.evaluate --config configs/research/motion_rollout.yaml \
  --checkpoint outputs/research/motion-rollout/checkpoints/last.ckpt \
  --reference rgb --horizons 1 2 4 --feature-metrics --output outputs/research/rgb-rollout-val.json
```

`--reference persistence` 保持最后场景；`--reference future_rgb` 读取真实未来特征作诊断。后者使用未来观测，不能计入 forecasting 排名。

## 9. 恢复、归档与结论

```bash
python3 -m foresight.research.train --config configs/research/motion.yaml \
  --resume outputs/research/motion-explicit/checkpoints/last.ckpt \
  --output-dir outputs/research/motion-explicit
```

checkpoint 保存模型与训练配置、优化器及训练步数。epoch 中途恢复可能重读该 epoch 开头部分样本，不保证逐样本位级重现；正式预算比较记录实际 global_step。恢复使用一致的数据、模型与优化协议；不得把阶段之间的初始化当成同一运行 resume。保存完整输出目录、配置、commit、数据预处理证据、样本 manifest、GPU/精度/全局 batch、运行耗时、权重 hash。checkpoint 选择规则与预算必须在 test 前确定。

实验 Agent 最终提交：

1. 每个命题的结论：支持／反对／证据不足，附对应匹配对照。
2. 全部方法与 seeds 的 val/test 指标；标注百分点、样本数、类别稀疏性。
3. 按 horizon 的误差与 AP 曲线，前景/背景区分；必要时按序列分组。
4. 实际训练的模块、参数量、预算，以及未能严格匹配的因素。
5. 失败、负结果、运行错误和协议变更；下一轮只针对已经定位的瓶颈。

模型损失变小、CPU 单测通过、合成训练完成，都不等于真实数据上的研究命题成立。文档中所有真实实验结果保持空白，待执行后填入。

## 10. 交付验证

2026-11-19：第一阶段提交树独立导出后运行完整测试，103 项通过，1 项跳过。环境为 Python 3.12 / PyTorch 2.15 CPU；跳过项需要通过 `FORESIGHT_ASSET_DIR` 显式提供真实 RGB 预训练权重。训练、评价、比较 CLI 已检查，覆盖合成 DSEC 数据、真实 Lightning 优化器恢复和 COCO 评价。真实 DSEC 训练、GPU/DDP 与研究指标由实验 Agent 按上述协议执行。
