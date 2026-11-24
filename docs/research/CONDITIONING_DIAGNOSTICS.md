# Event 条件分支：固定 checkpoint 的依赖与梯度诊断

更新日期：2026-11-24。实现分支：`codex/event-conditioning-diagnostics`。本实验延续 [当前方法与训练交接](DINOF_MOTION_CONDITIONING.md)，检查首轮 DSEC 结果中“特征误差下降，但检测 AP 没有稳定增加”的原因。实验使用已有 checkpoint，不训练新模型，不改变 Event 分辨率、模型参数或 scene loss。

## 1. 本轮要回答的问题

当前 Event 分支将逐 bin CNN 特征作为 memory，由 RGB future hidden 生成的 query 读取，然后在 DinoF 第六个完整 block 后注入残差。分支包含 `q + CrossAttention(q, E)`，因此存在不依赖 Event 内容的 RGB query 路径。训练只使用未来 RGB PCA latent 的 SmoothL1，没有独立的运动监督。

首轮用户提供的报告中，ordered joint 开启条件分支时 val latent L1 为 0.4405、AP 为 0.3727；关闭整条分支后 L1 为 0.4419、AP 为 0.3748。这个对比说明整条条件路径改变了预测，但不能单独证明模型利用了 Event，也不能据此判断 Event 本身没有价值。报告来自训练服务器；本机未独立读取其 checkpoint、完整 JSON 或训练日志。

本轮固定 checkpoint、RGB、样本、目标 latent 与 detector，通过干预 Event 和分支路径回答：

1. 正确 Event 与错配 Event 是否产生不同预测，正确 Event 是否更有利于未来 latent 和检测？
2. 模型是否对区间内时间 bin 的内容顺序敏感？
3. 分支作用主要来自 Event memory，还是直接的 RGB query 路径？
4. 当前 checkpoint 上，scene loss 对 CNN、时间 embedding、Q/K/V 与输出投影是否存在有限、非零的局部梯度？

本轮不直接回答高分辨率 Event 是否更好、encoder 是否已经学到可解码的物理运动、训练历史中是否始终存在有效梯度。它也不替代收敛曲线或冻结特征的运动读出实验。

## 2. 版本与服务器资产

诊断分支基于已推送的 `codex/event-dinof-motion-conditioning@3eed669cc96e24986540908d668b38dd09ffae87`。用户报告中的服务器提交 `84b6024`（周期 checkpoint）和 `9902442`（Event cache 覆盖）不在这个远程基点中。本分支按已有实现从在线 HDF5 读取历史 Event，不假定具备服务器私有的缓存入口。正常 recipient 与错配 donor 共用 `ResearchDSECDataset.read_history_events()`；服务器若移植缓存，两条路径必须同时经过这个入口，验证在线/缓存内容一致后才可比较。

服务器执行时应保留原训练 checkout 与产物，另建诊断工作目录。记录原训练 revision、诊断 revision、checkpoint 文件哈希及两者差异，不能把诊断分支 revision 写成原训练 revision。若服务器曾修改网络结构或 checkpoint schema，应先核对差异；本诊断不放宽权重与科学配置的兼容性检查。

需要服务器已有的资产：

- ordered joint checkpoint 与其原始归档配置；建议同样诊断 ordered warmup，以区分两个阶段的现象。
- 原 `dinof.pt`、共同冻结 detector，以及与训练一致的 DSEC RGB、Event HDF5、检测标签和配准信息。
- 首轮评价使用的配对 manifest；如果尚未保存，先按同一筛选配置重建并核对完整 inventory。

加载 checkpoint 时使用原归档配置，保留模型维度、输入分辨率、损失、batch/优化设置等完整科学配置，不为单卡诊断随意更改配置来绕过兼容检查。归档中若包含服务器新增字段，应先确认它们的语义与读取兼容性，不能直接丢弃。资产路径需要迁移时，记录路径映射并核对内容哈希。独立的诊断 CLI 参数只覆盖 loader batch、worker 与样本范围；设备类型继承 `config.runtime.accelerator`，GPU 可见性通过 `CUDA_VISIBLE_DEVICES` 控制，不更新训练参数。

## 3. 配对干预

所有干预使用同一 checkpoint。`correct` 必须存在并作为差值参照；同一样本的 RGB、GT 和目标时间不随干预改变。

| 名称 | 改动 | 解释边界 |
|---|---|---|
| `correct` | 原始历史 Event，正常条件分支 | 当前 checkpoint 的正常推理结果 |
| `mismatched` | 用完整评价 inventory 中其他 sequence 的历史 Event 替换当前 Event | 检查内容依赖；同时改变场景、活动量等统计，不是纯时间干预 |
| `bin_reverse` | 每个历史区间内反转 raw bin 内容，固定三个区间顺序和全部时间 embedding 槽位 | 检查区间内顺序敏感性 |
| `bin_shuffle` | 每个样本使用可复现、非恒等的 bin 排列，作用于各区间的 raw bin 内容 | 检查另一种区间内顺序扰动；相同 seed 和 sample ID 得到相同排列 |
| `zero` | Event voxel 全零，仍运行 CNN、时间 embedding 和条件读取 | 辅助压力测试；bias 和位置 embedding 仍在，结果不等同于完全无 Event 路径 |
| `query_only` | 置零 cross-attention 的输出，保留 query residual、FFN、motion head 和输出投影 | 分离直接 RGB query 路径；是同一已训练模型的干预，不是重新训练的 adapter baseline |
| `off` | 关闭整条条件注入路径 | 当前 checkpoint 的 RGB backbone 消融；不是原始 frozen RGB，也不是独立 RGB 微调 baseline |

错配 donor 映射在完整 inventory 上构建，再选择诊断子集和分批。它按 seed 和 sample ID 确定，与 Python hash seed、batch 大小和 inventory 遍历顺序无关。必须至少有两个 sequence；donor 来自其他 sequence，允许不同 recipient 重复使用同一 donor，不要求一一置换。这样 batch 为 1 仍能错配，且不会因相邻重叠窗口而用到几乎相同的事件。donor 只提供历史 Event，目标 RGB 与 GT 始终来自 recipient。

bin 干预发生在 CNN 和时间 embedding 之前，固定位置编码所表示的时间槽。不能把已经携带位置编码的整行 K/V 一起重排：attention 对这种 memory 行置换不敏感，不能用它检验时间信息。

保持输入为原配置的 `3 × 5 × 2 × 128 × 256`，保持原 CNN、DinoF、标准化/PCA 和 detector。分辨率实验会同时改变输入信息和计算，本轮不混入。

## 4. 指标与记录

`report.json` 的格式标记为 `event-conditioning-diagnostics-v1`，评价类型为 `conditioning_intervention_diagnostic`。它在全部样本处理完成、状态哈希不变检查通过后生成；只有目录和部分 JSONL，不代表本轮已经完成。

| 字段/产物 | 含义 |
|---|---|
| `variants.<name>.detection` | 共同冻结 detector 的 AP、AP50、AP75 与类别指标；`--skip-detection` 时为 null |
| `variants.<name>.features` | 对未来 GT 的全图/前景/背景 L1、SmoothL1，以及前景 patch 占比；区域指标按 patch 数聚合，空区域为 null |
| `prediction_change_vs_correct` | 干预预测与正确 Event 预测之间的直接 L1 差，按全图/前景/背景分开；不是两次 GT loss 相减 |
| `encoded_events_*`、`memory_*` | 位置编码前的 Event 特征、加入时间位置后的 memory 的 RMS 和相对 correct 的 L1 差 |
| `attention_*`、`motion_*`、`update_*` | cross-attention 输出、motion head 输出、注入残差的 RMS 与相对 correct 的 L1 差 |
| `update_to_hidden_l2_ratio` | 同一样本的残差范数相对被注入 future hidden 的范数 |
| `attention_to_query_l2_ratio` | attention 输出相对直接 query 路径的范数 |
| `encoded_temporal_variation_rms` | 加入位置编码之前 Event 特征沿时间轴的变化，避免把位置 embedding 差异误当作 Event 内容差异 |
| `protocol_timing` | 实际 Event 历史、RGB 输入支持范围、cutoff 到目标的时间差，各自记录 min/mean/max，单位 ms |
| `<variant>.samples.jsonl` | 逐样本特征、直接预测差异、分支统计、donor ID 与因果时间元数据 |
| `<variant>.predictions.jsonl` | 检测框、类别与分数，供配对分析；跳过检测时不生成 |
| `gradients.jsonl` | 每个 probe 样本的 scene loss、模块参数/梯度 RMS、非零比例、断开梯度张量数量和 Event 输入梯度 RMS |
| `run.json`、`manifest.json`、`donor_map.json` | 完整配置、版本和资产哈希、实际 recipient 清单、完整 donor 映射与 seed |

梯度记录将 PyTorch 打包的 cross-attention Q/K/V 投影拆开统计。`gradient_rms: null` 表示该组没有连接到当前 loss；非空但全零与此不同。probe 中 Predictor 权重固定，仍对其后六层的输入求导，以免分配完整 Predictor 参数梯度。实现使用 `autograd.grad`，不写入参数 `.grad`、不调用 `optimizer.step`；参数的原 `requires_grad` 状态在 probe 后恢复，不保存新 checkpoint。

`encoded_temporal_variation_rms` 对全部 15 个 Event tokens 相对时间均值计算，包含三个区间之间和区间内的变化，不是仅针对区间内五个 bins 的变化量。`protocol_timing` 描述实际模型输入；原检测元数据中的候选窗口跨度单独标为 `inventory_history_ms/actual_inventory_history_ms`，不能把约 1950 ms 的 inventory 历史误写为四帧模型实际输入。

聚合报告中的 `diagnostic_means` 是逐样本分支统计的均值，与区域误差按 patch 数加权的规则不同。`summary.md` 仅展示数字与解释边界，不用固定阈值自动判断根因。AP 存储为 0–1，摘要显示为百分数；所有 ΔAP 的方向均为干预减 `correct`，单位为百分点。

## 5. 如何解释结果

先看 Event 替换是否影响最终预测，再看这些变化是否有用，最后结合梯度判断当前路径是否可训练。

| 观测组合 | 支持的判断 | 尚不能判断 |
|---|---|---|
| `correct` 与 `mismatched` 的预测几乎相同，但 `off` 明显不同 | 当前 checkpoint 对 Event 内容依赖较弱；RGB query 路径可能承担主要修正 | 不能仅凭此区分 encoder 没编码信息还是融合没使用信息 |
| `correct` 与 `query_only` 接近，二者都与 `off` 不同 | 直接 query 路径解释了大部分当前分支效果 | 不能证明训练过程中 Event 从未影响 Predictor |
| `correct` 明显优于 `mismatched`，但 reverse/shuffle 影响小 | Event 内容有贡献，尚缺少细粒度顺序贡献证据 | 不等于所有时间信息都无用；三个 interval 顺序没有被打乱 |
| 干预显著改变 latent，但正确 Event 未改善误差或 AP | 分支使用了 Event，但当前使用方式缺少任务收益 | 输出敏感性本身不是有效运动表征 |
| 正确 Event 改善 latent 误差，但 AP 更差 | 特征回归收益没有传递到固定检测读出 | 需要进一步区分定位、分类和空间区域；不能直接宣布 loss 是唯一原因 |
| 只有 `zero` 变化大，真实跨 sequence 替换影响小 | 对分布外零输入敏感，正常 Event 内容依赖仍可能较弱 | 零输入敏感性不能替代真实内容消融 |
| 某模块在局部 probe 中梯度缺失、全零或非有限 | 存在应进一步定位的当前样本/当前 checkpoint 路径问题 | 不能推广为全部训练过程断梯度 |

平均 GT latent L1 相同并不代表预测相同；直接预测差异必须单独报告。注入残差小也不必然意味着没有作用，后续网络可能放大或抑制它。不能用固定阈值把一个范数自动判为“学到了运动”或“没有学习”。

`gradient probe` 使用当前 checkpoint 和选定样本，按归档配置计算原 scene loss（首轮配置为 SmoothL1）的局部导数，并应用原 `scene_weight`，不调用 optimizer step，不更新权重。表中 L1/SmoothL1 是未乘 `scene_weight` 的评价值。它不是原训练日志，也不是历史参数更新记录。非零梯度只能表明当前位置存在可微敏感性，不能证明训练充分、优化方向有效或编码了物理运动；跨模块的原始梯度范数也受参数量与尺度影响。

先在小批配对 val 样本验证数据、干预和报告完整性，再在完整 val 上分析。模型或预算的后续选择只使用 val；确定方案后再做最终 test。diagnostic AP 是同一 checkpoint 的受控干预，不能将其中最好的干预作为独立训练的新 baseline，也不能用单 seed 的细小差异声称稳定收益。

## 6. 服务器执行顺序

在具备原训练环境、DSEC 和资产的机器上，从现有代码仓库创建独立诊断目录：

```bash
git fetch origin codex/event-conditioning-diagnostics
git worktree add ../event-conditioning-diagnostics origin/codex/event-conditioning-diagnostics
cd ../event-conditioning-diagnostics
```

诊断是单进程程序，使用一张可见 GPU，不通过 `torchrun` 或 DDP 启动。以下四个路径由服务器实际产物替换；`DIAG_MANIFEST` 可以直接指向原始完整 val 评价 JSON，也支持含 `split`、`sample_ids` 的 JSON 或纯 sample ID 列表。

```bash
DIAG_CONFIG=/absolute/path/to/ordered-joint/config_train.yaml
DIAG_CHECKPOINT=/absolute/path/to/ordered-joint/checkpoints/last.ckpt
DIAG_MANIFEST=/absolute/path/to/original-full-val.json
DIAG_OUTPUT=/absolute/path/to/new-conditioning-diagnostics
```

新输出目录必须不存在。小批诊断与完整 val 使用不同目录；失败后保留原目录以便排查，重跑另用新目录。

先在服务器执行合成数据模型测试；这些命令包含模型前向与局部梯度测试，不是训练任务，本机不执行：

```bash
python -m pytest -q tests/test_conditioning_diagnostic_*.py \
  tests/test_research_conditioning.py tests/test_research_evaluate.py
```

运行 128 个 recipient 的 val 诊断，并对前 8 个样本执行当前 checkpoint 的局部梯度 probe：

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.research.diagnose_conditioning \
  --config "$DIAG_CONFIG" \
  --checkpoint "$DIAG_CHECKPOINT" \
  --manifest "$DIAG_MANIFEST" \
  --split val \
  --output-dir "${DIAG_OUTPUT}-val128" \
  --max-samples 128 \
  --gradient-samples 8 \
  --loader-batch-size 1 \
  --num-workers 0 \
  --seed 42 \
  --variants correct mismatched bin_reverse bin_shuffle zero query_only off
```

默认 `--gradient-samples 0` 不计算梯度；probe 逐样本执行，运行期间没有 optimizer step。`--loader-batch-size 1` 仅覆盖诊断 DataLoader，不更改 checkpoint 归档的训练 batch/累积步数。默认 worker 为 0，避免不必要的 HDF5 多进程读取；不传 `--loader-batch-size` 时使用原配置的 batch。

`--max-samples` 取选定 manifest 的前 N 个 recipient，通常不是均衡或随机子集，因此只用于检查流程。donor pool 仍是完整 manifest，可从未被选中的 recipient 样本取 donor。若要快速检查特征路径，可以增加 `--skip-detection`；这种输出没有 AP，不能据此分析检测收益。

确认 `correct` 诊断前向与原正常前向一致、权重/缓冲状态未改变、donor 全部跨 sequence、输入与输出没有非有限值后，运行完整 val。首批一致性与结束时权重哈希由 runner 检查；失败会中止，不产生完成标记 `report.json`。

```bash
CUDA_VISIBLE_DEVICES=0 python -m foresight.research.diagnose_conditioning \
  --config "$DIAG_CONFIG" \
  --checkpoint "$DIAG_CHECKPOINT" \
  --manifest "$DIAG_MANIFEST" \
  --split val \
  --output-dir "${DIAG_OUTPUT}-full-val" \
  --gradient-samples 0 \
  --loader-batch-size 1 \
  --num-workers 0 \
  --seed 42
```

完整运行默认包含七个干预，不设置 `--max-samples`。`scope.is_full_inventory` 只相对选定 manifest 为真；需要另外核对该 manifest 是首轮完整 val，而不是曾经截断的清单。首次完整 `correct` 结果应与同 checkpoint、同 inventory 的已有正常评价核对；若不一致，先定位版本、在线读取、配置或精度差异，不解释其他干预。

ordered warmup 使用其自己的原配置和 checkpoint，沿用同一完整 val manifest 与 seed，换一个新输出目录。若诊断 pooled checkpoint，务必保留它原本的 `event_pooled` 模式；池化机制本身使区间内顺序干预预期影响较小，不能把这一现象当作 ordered 分支的证据。runner 只接受 `stage: conditioning`、单步、`event_ordered/event_pooled` 的 checkpoint，不接受 RGB-only arm。

runner 自动生成 `summary.md`。如果只需在无模型依赖的机器上重新生成摘要，可运行标准库入口，并使用一个新的输出文件名：

```bash
python3 -m foresight.research.conditioning_report \
  "${DIAG_OUTPUT}-full-val/report.json" \
  --output "${DIAG_OUTPUT}-full-val/summary-review.md"
```

该入口不导入 torch、numpy 或 Lightning，拒绝覆写已有 Markdown。摘要不代替逐样本分析，也不计算跨 seed 的统计显著性。

## 7. 交付与后续分析

本轮交付诊断代码、标准库摘要工具和文档。服务器 agent 执行诊断并分析结果；不需要启动训练或修改训练日程。

服务器交回原始报告、逐样本干预记录、完整配置与 revision、checkpoint/资产哈希、inventory 信息、控制台日志，并明确是否为完整 val。报告须同时保留正常条件和所有干预，不能只选择 AP 最好的结果。若分析发现需要扩大训练或改模型，先基于本轮证据提出独立实验，不把改动并入同一组固定 checkpoint 结果。

## 8. 验证状态

开发机器不具备 torch、GPU、DSEC 或生产权重。本机通过 29 项纯 Python 协议/摘要测试、9 个 Python 文件的 AST 语法检查，以及 CLI 帮助和 diff 检查。模型干预、局部梯度和 runner 集成测试已编写，需在服务器环境执行。没有执行本轮 DSEC 诊断、生产模型前向/反向、optimizer step、训练或完整集 AP；本文不包含新的实验结果。
