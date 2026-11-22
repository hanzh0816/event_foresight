# 真实未来事件与场景转移：诊断实验交接

本实验只评价已有 checkpoint，定位当前 Event–RGB 运动预测通路的瓶颈。先完成单步验证集诊断，再根据证据决定是否开展多步诊断或重新训练。

## 1. 研究上下文与本次问题

研究目标是：用预训练 RGB 模型承载场景知识，用事件编码器 `G` 提取区间运动表示，预测未来运动后，通过场景转移器 `T` 推进未来场景 latent，最后用共同检测头评价驾驶场景的未来检测。完整研究背景见 [研究方案](event-wm-plan.md)，原有实验设计见 [实验交接](EXPERIMENT_HANDOFF.md)。

上一轮实验 Agent 反馈：冻结 DINO-Foresight predictor 的验证集 AP 约为 **36.64**，显式运动预测通路约为 **19.26**；融合增益未能跨 seed 稳定复现，运动显式监督也未超过匹配的自由条件或辅助监督对照。这些数字是已有反馈中的上下文，不是本分支重新运行或核验的结果。原报告使用 0–1 AP，这里乘以 100 展示。

目前无法仅凭这些结果判断：是 `G/T` 不能充分解释实际发生的场景变化，还是历史到未来运动的预测不准确。先回答：

> **将真实未来区间的事件输入同一个 `G/T`，在同一 detector、样本和时距下，未来场景能够达到多少 AP？与预测运动输入相比差多少？**

这次固定权重、读出与数据，只切换转移条件或场景来源，不新增训练，不修改已有训练结果。真实未来事件路径属于已观测变化的恢复诊断，不能列入只使用历史的 forecasting 排名，也不构成数学意义的性能上界。

## 2. 代码、资产与执行位置

- 本分支：`codex/event-wm-transition-diagnostic`，基于公开代码 `4a7dc5e`（`codex/event-wm-motion-rollout`）。
- 服务器实验分支反馈为 `codex/wm-run`、`e618f16`；该版本及其运行结果没有包含在本分支基线中。不要覆盖服务器实验目录，也不要假定其修改已经合并。
- 优先使用服务器已有的 **M-explicit、Event 输入、冻结 T** checkpoint。第一次诊断使用原实验选定的 checkpoint，不重新按本次诊断结果挑选 checkpoint。
- 需要已有 DINO-Foresight export、共同 detector export 和 DSEC 数据。直接复用上一轮资产，不需要重新导出 detector，也不依赖服务器的 `configs/research/b200/`。
- 模型结构和研究配置从 motion checkpoint 恢复；原始教师训练 checkpoint、初始化来源 checkpoint 不需要存在于当前路径。CLI 提供的资产和数据路径用于服务器间迁移。
- 入口严格恢复模型与因果协议；评估 batch size、workers、device、precision 在命令行设置，不修改训练 resume 的规则。若 checkpoint 格式不兼容，应保留报错并核查版本，不能跳过权重或结构检查。

可在服务器现有仓库旁建立独立 worktree。下面的路径是占位符，执行前换成实际路径；使用显式 GitHub URL，避免假定服务器 `origin` 指向 GitHub。

```bash
git -C /abs/event_foresight fetch https://github.com/hanzh0816/event_foresight.git codex/event-wm-transition-diagnostic
git -C /abs/event_foresight worktree add --detach /abs/wt/wm-transition-diagnostic FETCH_HEAD
cd /abs/wt/wm-transition-diagnostic
git rev-parse HEAD
```

激活上一轮可用的 Python 环境，确认解释器从当前 worktree 导入代码；若环境以 editable 方式绑定了旧仓库，在新 worktree 重新安装。沿用已验证的 detector export。共享 GPU 环境下保留 `OMP_NUM_THREADS=1`，避免 DataLoader worker 的 CPU 线程过度竞争。

```bash
export OMP_NUM_THREADS=1
python3 -m pip install --no-deps -e .
python3 -m foresight.research.diagnose_transition --help
```

## 3. 同一批样本上的路径

记最后历史场景为 `z₀`，历史最后一个事件区间表示为 `m₀`，真实未来场景为 `z*h`，第 h 个真实未来区间的事件表示为 `m*h = G(E*h)`。所有路径使用 checkpoint 中同一个 RGB encoder/PCA、事件教师、相同目标和共同 detector。

| 报告路径 | 场景生成方式 | 使用未来观测 | 作用 |
|---|---|---|---|
| `rgb_predictor` | 原 RGB predictor 从历史预测；多步递归 | 否 | 已有成熟预测能力参考 |
| `scene_persistence` | 每一步复制 `z₀` | 否 | 场景不变化参考 |
| `true_future_rgb` | 直接读出真实未来 `z*h` | 是，RGB | 检查 detector 对真实目标特征的读出表现 |
| `true_future_event` | `ẑh = T(ẑh−1, m*h)`，起点为 `z₀` | 是，Event | 给定真实变化观测时的场景恢复质量 |
| `last_motion` | `ẑh = T(ẑh−1, m₀)` | 否 | 保持最后观测运动的参考 |
| `predicted_motion` | checkpoint 原有历史 → 预测运动 → `T` 路径 | 否 | 实际运动预测通路 |
| `true_state_true_event` | `z̃h = T(z*h−1, m*h)` | 是，Event；第二步起还使用未来 RGB 起点 | 多步时分离单次转移质量与预测状态反馈影响 |

最后一项仅在多步诊断出现；第一步与 `true_future_event` 一致。多步 `true_future_event` 的起点在首步之后始终是自己的预测，不能在每一步偷偷换成真实场景。

第一轮优先使用冻结 T 的 M-explicit。入口也允许 Explicit 的联合适配 checkpoint：此时所有运动条件路径使用该 checkpoint 实际推理的 adapted T，目标运动仍由冻结教师 G 产生；报告分别记录 active T 和冻结教师的权重身份。不要把联合适配结果与冻结 T 结果当成只改变运动输入的对照。Direct、Auxiliary、RGB-motion 或 copy-last checkpoint 不用于本入口的主诊断，避免改变“历史 Event 预测的运动与同一个事件教师目标”这一问题。

## 4. 数据和指标必须共同

整个诊断使用一次共同 DataLoader 遍历，同一组历史、未来目标、标签、检测头和精度设置，不分别构造各路径的数据集。

- 保持原 checkpoint 的 RGB/Event 分辨率、历史长度、时间 bin、几何配准说明和官方 split。未来事件严格按真实目标区间读取。
- 默认 `val`、单步。先使用原评估 manifest，可直接传包含 `split` 和 `sample_ids` 的旧评估 JSON；若不指定，则按本次有效窗口协议生成共同集合。
- `steps=4` 使用四步共同 inventory，选择 horizons 1/2/4。它与一步 inventory 可能不同，不能直接对比两张表的 AP@1。多步比较以同一次报告内的路径为准。
- 真实未来事件、RGB 和真起点只进入各自诊断分支。`predicted_motion`、`rgb_predictor`、`last_motion` 和 `scene_persistence` 的生成只用历史。
- 配准仍采用 checkpoint 中记录的实际数据前处理。上一轮反馈为 DSEC-det RGB remap 到 raw Event 视角，存在无限深度近似造成的残余错位；不能把本实验视为消除了几何误差。

每条路径、每个 horizon 统一报告 **AP/AP50/AP75** 与 **全图、检测框前景、背景 latent L1**。必须用同一空间区域比较 L1，不能拿教师全图 L1 与 RGB 前景 L1 做差。前景按目标 GT 框覆盖的 latent 网格定义，是任务区域诊断，不等同于像素级语义 mask。

`predicted_motion` 和 `last_motion` 的 motion L1 均相对同一冻结 G 编码的真实未来运动。真实事件条件与自己构成的 motion 目标一致，其零误差不能证明表示有效；关键仍是它产生的场景 AP 和 latent L1。

detector 曾在 RGB predictor 输出上训练，因此真实未来 RGB 读出是分布诊断参考，不能宣称是所有方法必然低于的“上限”。真实事件路径也不是上界：学习出的条件可能补偿 T 的偏差，从而在某些指标上优于 G 的目标表示。

## 5. 执行命令

先用 32 个样本检查资产恢复、数据和各路径输出。该结果只用于排查流程，不能用于研究结论：

```bash
python3 -m foresight.research.diagnose_transition \
  --checkpoint /abs/M-explicit/checkpoints/last.ckpt \
  --dinof-checkpoint /abs/checkpoints/dinof.pt \
  --detector-checkpoint /abs/checkpoints/detector.pt \
  --data-root /abs/DSEC \
  --output-dir outputs/research/transition-diagnostic-seed42-smoke \
  --batch-size 8 --num-workers 8 --device cuda --precision bf16-mixed \
  --steps 1 --max-samples 32
```

检查 smoke 报告中的样本数、路径名、有限指标、checkpoint 身份和配准说明后，运行完整验证集。以下 manifest 必须来自同一 split、同一单步协议的旧评估；路径不可用时可以省略，由入口生成新 manifest，并记录样本数量变化。

```bash
python3 -m foresight.research.diagnose_transition \
  --checkpoint /abs/M-explicit/checkpoints/last.ckpt \
  --dinof-checkpoint /abs/checkpoints/dinof.pt \
  --detector-checkpoint /abs/checkpoints/detector.pt \
  --data-root /abs/DSEC \
  --output-dir outputs/research/transition-diagnostic-seed42 \
  --batch-size 8 --num-workers 8 --device cuda --precision bf16-mixed \
  --steps 1 --manifest /abs/outputs/research/A-val.json
```

不指定 `--max-samples` 会评价所选集合的全部样本；manifest 本身也可能只包含子集，因此是否覆盖本协议的完整验证集，以报告的 `sample_selection.covers_full_inventory` 和样本数为准。可加 `--save-predictions` 保存各路径检测结果，供配对可视化或后续统计。

输出目录必须尚不存在，不要提前创建。入口会自行创建，并最后写入 `report.json` 作为评价完成标记；失败重跑时换一个新目录，保留原目录用于排查。smoke 与正式评价使用不同目录。

只有单步结果需要进一步区分误差积累时，再做四步诊断。下面仍使用同一个 M-explicit checkpoint，评价其递归行为，不把它当成重新进行四步训练；如改用已训练的 R-rollout/R-joint，另建运行目录并记录训练时 rollout 长度。

```bash
python3 -m foresight.research.diagnose_transition \
  --checkpoint /abs/M-explicit/checkpoints/last.ckpt \
  --dinof-checkpoint /abs/checkpoints/dinof.pt \
  --detector-checkpoint /abs/checkpoints/detector.pt \
  --data-root /abs/DSEC \
  --output-dir outputs/research/transition-diagnostic-seed42-4step \
  --batch-size 8 --num-workers 8 --device cuda --precision bf16-mixed \
  --steps 4 --horizons 1 2 4
```

这一步使用新生成的四步 manifest；若已有同协议四步 manifest 可显式传入。先完成 val 诊断，不重复用 test 选择下一轮设计。

## 6. 如何根据结果作决定

| 观察结果 | 支持的定位 | 对下一轮工作的影响 |
|---|---|---|
| 真实未来事件路径的 AP 和前景 L1 仍明显差于 RGB predictor | 问题已存在于观测事件表示、场景转移或读出接口，而非只在运动预测器 | 优先诊断 G/T 的学习、容量和表示；暂缓扩大运动预测与 rollout |
| 真实未来事件路径明显改善，但预测运动路径很差 | 从历史预测教师运动目标，或预测条件分布与 T 的匹配，是主要损失来源之一 | 研究运动表示的可预测性、预测器与目标；同时用 last-motion 区分是否学到有效演化 |
| 真实未来 RGB 读出也明显低于 RGB predictor，或 AP 与 latent L1 改善方向不一致 | 共同 detector 的分布偏好可能影响结论 | 保留共同读出主表，先检查读出问题，再设计同协议读出适配；不能仅据 AP 归因于 WM |
| 多步真实起点转移尚可，但递归真实事件路径迅速恶化 | 预测场景反馈造成的累积误差是重要因素 | 优先研究多步状态分布与转移稳定性 |
| 真实起点转移本身随 horizon 也很差 | 即使重置起点，单次转移仍不足 | 先改善单次转移，不把问题全部归于 rollout |

这些是诊断分岔，不是互斥、唯一的因果结论。不存在预设“必须提升多少 AP 才通过”的门槛；先报告配对绝对值与差值、样本数和各 horizon 的趋势，再判断哪个差距值得下一轮实验。单 checkpoint 的结果不自动支持统计显著性，当前不新增多 seed 训练。

## 7. 产物与 Agent 反馈格式

入口写出 `report.json`、便于阅读的 `summary.md` 和实际样本 `manifest.json`；指定 `--save-predictions` 时额外写 `predictions.json`。报告整体标为 `transition_bottleneck_diagnostic`，各路径说明是否使用未来观测，不能混入原 `compare.py` 的正式 forecast 排名。

归档代码 commit、完整命令、输入 checkpoint SHA256、实际加载的 forecaster/冻结教师/active transition 权重身份、detector 身份、split、样本与时间协议、精度及配准依据。教师身份以 checkpoint 中实际嵌入的权重为准，不能只凭软链路径推断一致。

实验 Agent 返回以下内容，数值运行后填写：

1. **执行身份：**代码 commit、M-explicit checkpoint、实际权重 hashes、GPU/精度、完整命令，是否为完整 val。
2. **统一结果表：**各路径 AP/AP50/AP75、全图/前景/背景 L1、有效样本数；预测运动与保持运动的 motion L1。
3. **决定性差值：**预测运动 − 真实未来事件、真实未来事件 − RGB predictor、预测运动 − 保持运动；与报告的 `left − right` 方向一致，AP 差值明确使用百分点，L1 明确区域与方向。
4. **瓶颈判断：**观察到的证据、尚不能区分的因素，以及是否需要四步真起点/递归诊断。若执行多步，给出各 horizon 同表比较。
5. **下一步仅一项优先实验：**由上述分岔选择，并说明需要改变哪个模块、保持哪些条件。

此文档不预填诊断结论，也不要求实验 Agent 因为代码存在而继续扩大训练。先用已有 checkpoint 回答真实未来事件条件下的场景恢复质量。

## 8. 交付验证

本分支在 CPU 环境运行完整测试集：**159 passed, 1 skipped**；跳过的是需要通过 `FORESIGHT_ASSET_DIR` 显式提供真实预训练资产的可选测试。新增测试覆盖严格 checkpoint 恢复与路径迁移、未来观测不影响正式预测、多步递归与真起点区别、adapted T 一致性、合成 DSEC 数据上的完整评价和 manifest 重放。CLI `--help` 与 `git diff --check` 通过。

尚未使用服务器真实 checkpoint、完整 DSEC 或 GPU 运行本次诊断，因此本次交付包含实现与执行协议，不包含新的 AP 结果。实验 Agent 从第 5 节的 smoke 开始验证实际资产兼容性。
