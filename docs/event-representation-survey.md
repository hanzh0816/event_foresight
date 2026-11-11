# Event 特征提取、处理与对齐：已有文献调研汇总

整理与代码入口核查日期：**2026-11-11**。

本文以本机 `/root/papers` 的 Event 分类全文、事件编码与融合专题、事件生成与世界模型专题、驾驶与机器人专题为基础，按可复用的技术模块整理。主体覆盖已有专题中的相关工作；从这些论文的引用中补查的条目另作标记。每项给出论文与作者代码入口，代码状态以此次可访问页面为准；未运行各论文训练。

对当前项目，最直接的参考链是：**EST / RVT / EvT / F³ 的事件编码 → ScalE / GEP / REALM 的视觉知识迁移 → 历史状态与未来表示预测**。应用论文中的事件压缩、历史记忆、融合与辅助监督也值得复用。当前项目范围为训练可用配对 RGB/VFM，推理只用过去事件，无动作条件；下文同时保留 RGB–Event 融合和动作方法，并明确它们的输入接口。

## 阅读导航

- [1. 按模块快速选文献](#map)
- [2. 原始事件如何变成模型输入](#representations)
- [3. 编码器、时序状态与融合](#encoders)
- [4. RGB / VFM 知识迁移与特征对齐](#alignment)
- [5. 预测表示、未来生成与应用模块](#prediction)
- [6. 事件生成、重建与数据工具](#generation)
- [7. DSEC 的几何与时间对齐](#dsec)
- [8. 对当前项目的具体参考顺序](#project)
- [9. 本机来源与覆盖核对](#sources)

<a id="map"></a>
## 1. 按模块快速选文献

| 要解决的问题 | 优先参考 | 主要可复用部分 |
|---|---|---|
| 异步事件变成张量 | EST、RVT、ScalE、GEP | 分箱、极性处理、可学习聚合、预训练输入接口 |
| 保留细时间信息并压缩 token | EvT、FLEET、F³ | 活跃 patch、原始事件位置编码、latent cross-attention、稀疏 hash 与时间聚合 |
| 连续处理历史 | RVT、SSM、HMNet、RAMNet | 空间交互与时序状态分工、多时间尺度记忆、异步状态更新 |
| Event encoder 对齐冻结 RGB encoder | ScalE、GEP、REALM、Network Grafting | dense feature 蒸馏、空间关系约束、冻结后端复用 |
| 图像／文本语义迁移 | EventCLIP、EventBind、ECDP | 伪帧初始化、对比学习、全局与局部迁移 |
| 分割或深度知识迁移 | EventSAM、ESS、EvDistill、OpenESS、DepthAnyEvent | 任务相关蒸馏、语义／几何监督与 dense readout |
| 学习未来有用的事件特征 | F³、GEP、E-Motion | 未来事件目标、因果 latent 预测、视频生成先验 |
| RGB 与 Event 在线融合 | RAMNet、EFNet、DAGr、DMR、DRFuser | 共享状态、cross-attention、异步更新、任务相关特征／模态噪声分解 |
| 多模态策略中的特征接口 | FLEET、ECHO、E-VLA、Event-VLA、EventDrive、eNavi | 固定数量 token、轨迹记忆、门控适配器、未来辅助损失 |
| 去噪、补全、重建与合成 | EDFilter、EDR、E2VID、EvFRA、LongE2V、ESIM、v2e | 事件质量控制、视觉 decoder、训练数据生成 |

其中“语义对齐”“时空对齐”“多模态融合”解决不同问题。语义对齐使两路表示具有可比较的含义；时空对齐确定两路数据是否对应同一位置和时刻；融合则在推理时结合多路输入。后者的收益不能直接作为 Event-only 蒸馏的证据。

<a id="representations"></a>
## 2. 原始事件如何变成模型输入

原始事件写作 $e_i=(x_i,y_i,t_i,p_i)$，其中 $p_i\in\{-1,+1\}$。输入窗口应同时记录起止时间、实际事件数量和极性约定。固定时长窗口保留统一的物理时间尺度；固定事件数窗口使输入负载相对稳定，但持续时间随运动和亮度变化。若采用后者，时间归一化之后仍应保留实际窗口时长，才能比较速度和预测跨度。

| 表示 | 计算与保留的信息 | 代价或信息损失 | 对应参考 |
|---|---|---|---|
| 正／负计数图 | $C_p(x,y)=\sum_i\mathbf1[(x_i,y_i,p_i)=(x,y,p)]$ | 聚合便宜；窗口内先后顺序消失 | Event Steering、EventCLIP、GEP |
| Time surface | 每个像素／极性的最近事件时间，或相对当前时刻的指数衰减 | 保留新近性；多次触发数量与完整轨迹被压缩 | HOTS、HATS |
| Signed voxel grid | 时间轴分为 $B$ 个 bin，以符号极性累积；可做时间线性插值 | 保留粗时间顺序；同位置正负可能抵消 | EST、E2VID、RAMNet、ScalE |
| 正负分离 voxel / histogram | 每个 bin 的 ON/OFF 独立计数，通常为 $2B\times H\times W$ | 避免极性抵消；bin 内仍被聚合 | RVT、EvT、Text-to-Events |
| Event pseudo-image | 将计数、极性或 mask 映射为三个通道，适配图像 ViT | 三通道定义随方法不同；并不等于真实 RGB | GEP、EventCLIP、EventBind |
| 原始事件集合／图 | 对 $(x,y,t,p)$ 编码，再用集合聚合、图邻域或 cross-attention | 时间信息较完整；计算受事件率和采样策略影响 | AEGNN、DAGr、FLEET、F³、EDR |
| 重建图像或生成 latent | 先由事件得到强度图或 decoder latent，再接视觉模型 | 引入重建偏差、计算及生成先验 | E2VID、EvDiff、EvFRA |

常见 signed voxel 的时间插值可写作：

$$
\widetilde t_i=(B-1)\frac{t_i-t_a}{t_b-t_a},\qquad
V_b(x,y)=\sum_{i:(x_i,y_i)=(x,y)}p_i\max(0,1-|b-\widetilde t_i|).
$$

这是便于比较的示意定义；具体实现有硬分箱、不同归一化、空间插值和正负分离等变体。**三通道 voxel、三通道计数伪帧和三帧历史是三种不同输入**：例如 ScalE 的 $B=3$ voxel 与 GEP 的正计数／活跃 mask／负计数不能互换。公式及表征分类参考 [EST](https://arxiv.org/abs/1904.08245) 和[事件视觉综述](https://arxiv.org/abs/1904.08405)。

对特征学习影响较大的预处理选择还有：去除热像素／背景噪声、计数截断或 log 压缩、空窗口处理、坐标映射后插值、训练时两模态共用裁剪与翻转。它们应与 encoder 一起记录，而不是只比较 backbone 名称。

<a id="encoders"></a>
## 3. 编码器、时序状态与融合

本节先介绍已有编码专题的主要方法，再列论文引用中的补充基线。下列 `sources/*.txt` 均指本机 `notes/surveys/event-rgb-vjepa2-design-2026-10-10/` 下的归档全文。

### EST：可学习的事件聚合前端（ICCV 2019）

- **输入与处理：**Event-only。把事件看作时空脉冲，按 measurement function（计数、极性或时间）、aggregation kernel 与离散网格构建 Event Spike Tensor；保留空间、时间、极性四个维度。用 MLP 学习 kernel，再接普通卷积网络，训练时联合优化下游任务。
- **时序机制：**在输入窗口内部表达时间；EST 本身不是跨窗口 recurrent memory。论文还讨论部署时把学习到的核转成 lookup table。
- **可借鉴之处：**将“事件如何变成 tensor”纳入可学习模块；在固定后端下比较 signed voxel、polarity-separated histogram 和 learned aggregation，能够区分表示收益与网络收益。
- **边界：**原验证包括识别与运动估计；不能直接据此前端效果判断未知未来预测能力。
- **链接：**[论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html) · [官方 GitHub](https://github.com/uzh-rpg/rpg_event_representation_learning)。本机全文：`sources/est-2019.txt`（上述目录）。

### RAMNet：不同传感器异步更新同一个状态（RA-L 2021）

- **输入与网络：**Event + intensity frame 的单目深度估计。事件先构造 5-bin signed voxel，Event / image 使用独立 U-Net 式编码器；多尺度 ConvGRU 将当前模态特征写入共享隐藏状态，decoder 从各尺度状态读取深度。
- **时序／对齐：**按实际输入到达顺序更新，并不要求每个更新点同时具有图像和事件。这里解决的是不同模态更新频率与状态融合，不能替代几何校准。
- **训练／推理：**使用深度标签训练；论文主要多模态设置推理仍用事件与图像。它允许参考异步 memory 设计，但不自动满足训练 RGB、推理 Event-only 的设定。
- **可借鉴之处：**将连续观察累积成可随时读取的场景状态；统一状态与模态专属更新函数的分工清楚。
- **链接：**[论文](https://arxiv.org/abs/2102.09320) · [作者项目](https://rpg.ifi.uzh.ch/RAMNet.html) · [官方 GitHub](https://github.com/uzh-rpg/rpg_ramnet)。本机全文：`sources/ramnet-2021.txt` §III-A/B/D。

### EvT：活跃 patch 与固定规模 latent memory（CVPR Workshops 2022）

- **输入与处理：**Event-only。按极性分别建立 $B$-bin 计数直方图，执行 $\log(1+F)$ 压缩，切成非重叠 patch，仅保留活跃像素比例达到阈值的 patch。
- **时序机制：**固定数量 latent memory 为 query，变长活跃 patch 为 key/value；cross-attention 读入事件，再由 self-attention 更新 latent；当前输出与已有 memory 相加，累积多窗口信息，最后用于识别。
- **可借鉴之处：**事件数量和活跃区域可变，而记忆 token 数量有界；适合比较“稠密空间网格”与“有限 latent token”压缩。
- **边界：**原模型是分类器，latent 不天然逐位置对齐 RGB patch。论文在活跃 patch 太少时扩展时间窗口；用于 forecasting 时应固定输入截止点，扩窗不能取得截止之后的事件。
- **链接：**[论文](https://openaccess.thecvf.com/content/CVPR2022W/ECV/html/Sabater_Event_Transformer._A_Sparse-Aware_Solution_for_Efficient_Event_Data_Processing_CVPRW_2022_paper.html) · [官方 GitHub](https://github.com/AlbertoSabater/EventTransformer)。本机全文：`sources/evt-2022.txt` §3.1–3.2。

### AEGNN：按新事件局部更新特征（CVPR 2022）

- **输入与处理：**Event-only。将事件组织为稀疏时空图，用 graph neural network 提取表示；新事件改变局部邻域，只重新计算受影响的激活。
- **训练／推理：**官方实现训练前离线生成事件图，提供将图模型转为异步稀疏执行的工具；在识别与检测数据上评估。它的异步性来自网络计算的增量更新。
- **可借鉴之处：**如果研究目标包含随事件到达持续更新状态，AEGNN 比 dense encoder 后裁剪 token 更直接；可以借鉴缓存、中间激活更新与局部传播。
- **边界：**需要 graph 构建与异步算子支持，不能仅按 theoretical FLOPs 推断实际吞吐；原任务是当前感知。
- **链接：**[论文](https://openaccess.thecvf.com/content/CVPR2022/html/Schaefer_AEGNN_Asynchronous_Event-Based_Graph_Neural_Networks_CVPR_2022_paper.html) · [官方 GitHub](https://github.com/uzh-rpg/aegnn)。本机依据：`prior-work.md`、`source_ledger.json`；该库此前为摘要／项目页初筛，本轮重查官方实现。

### RVT：将空间交互与时序记忆分开（CVPR 2023）

- **输入与处理：**Event-only。窗口内正、负计数各有 $B$ 个 bins，得到 $[2,B,H,W]$ 后展平为 $[2B,H,W]$，供 2D convolution 使用。
- **网络与时序：**多阶段下采样提取多尺度特征；每阶段使用局部 block attention 与 dilated grid attention 进行空间交互，再用逐空间位置的 LSTM 聚合历史。LSTM 不承担邻域空间混合，其输出同时用于检测和下一步状态更新。
- **训练／推理：**原论文训练目标为目标检测，推理只需事件与历史 state；带持续历史的检测性能不等价于未来 detection 性能。
- **可借鉴之处：**是规则网格、可接 dense task head 的强 Event encoder 参考。可以保留空间特征并增加 RGB 对齐头、future predictor；需要报告 state reset、warm-up、窗口与训练截断规则。
- **链接：**[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html) · [官方 GitHub](https://github.com/uzh-rpg/RVT)。本机全文：`sources/rvt-2023.txt` §3–3.2。

### SSM for Event Cameras：处理训练与推理频率变化（CVPR 2024）

- **输入与网络：**Event-only，沿用张量化输入和多阶段空间 backbone，使用 S4 / S4D / S5 等 state space module 承担时间聚合；官方 SSM-ViT 使用 S5 代替 RNN。
- **时序机制：**通过连续状态模型的离散化与 timescale 参数适配不同更新间隔，分析并处理跨频率的 aliasing；S5 训练可使用 parallel scan，推理携带状态。
- **可借鉴之处：**当训练事件窗口与部署更新周期不同时，提供比直接替换 GRU 更完整的参考。应显式比较 fixed frequency 与 changed frequency 下的表示、检测、预测表现。
- **边界：**论文验证围绕事件检测；不是“加入 SSM 就能任意变频泛化”的结论，也未证明同一状态足够支持未知未来生成。
- **链接：**[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Zubic_State_Space_Models_for_Event_Cameras_CVPR_2024_paper.html) · [官方 GitHub](https://github.com/uzh-rpg/ssms_event_cameras)。本机全文：`sources/ssm-2024.txt` §3.1–3.2。

### DAGr：事件图更新图像帧之间的检测（Nature 2024）

- **输入与网络：**主要汽车视觉设置为 RGB + Event；图像提供较低频的视觉上下文，事件图分支在图像帧间更新目标检测。
- **可借鉴之处：**把“较稳定的场景上下文”与“新到达事件改变哪些状态”分开，提供 Event–RGB 多尺度融合及低延迟状态更新参考；官方实现同时包含 DSEC-Det 的数据组织、训练与帧间评估入口。
- **边界：**帧间检测使用已经到达的新事件，是当前感知；如果目标在 $t+\Delta$，期间事件在 $t$ 时仍未知，不能把此结果计为 $t$ 时刻的未来预测。论文的多模态推理也不满足 Event-only 推理条件。
- **链接：**[论文](https://www.nature.com/articles/s41586-024-07409-w) · [官方 GitHub](https://github.com/uzh-rpg/dagr)。本机依据：`prior-work.md`、`/root/papers/notes/surveys/event-world-action-research/literature_matrix.md` §5；本轮重查官方代码说明。

### EFNet：曝光区间事件与图像特征的选择性融合（ECCV 2022）

- **输入与处理：**模糊图像 + 曝光期间的事件。SCER（Symmetric Cumulative Event Representation）围绕曝光中心在两侧累积事件，编码相对亮度变化。
- **融合机制：**两阶段去模糊网络在多尺度加入 event-image cross-modal attention：image feature 为 Q，event feature 为 K/V，attention 主要沿 channel 维计算；event mask gated connection 选择编码／解码特征进行阶段间融合。
- **可借鉴之处：**跨模态选择、噪声抑制、不同尺度融合；其 channel attention 与逐空间 token attention 是不同设计。
- **边界：**推理仍需图像和目标曝光区间事件，任务为重建／去模糊。SCER 的双侧累积不能直接作为相同中心时刻的因果预测输入。
- **链接：**[论文／作者项目](https://ahupujr.github.io/EFNet/) · [官方 GitHub](https://github.com/AHupuJR/EFNet)。本机全文：`sources/efnet-2022.txt` §3.3–3.4。

### 本机引用或 baseline 中出现的补充编码方法

以下工作在当前库内是引用、比较项或组件线索，阅读深度不同于上面的已归档方法全文；本轮据官方页面补核。

| 方法 | 事件提取／处理与可参考点 | 训练、推理与证据边界 | 论文与 GitHub |
|---|---|---|---|
| **HMNet，CVPR 2023** | attention 将稀疏事件写入 latent memory；多层 memory 按不同频率运行，快层提取局部动态、慢层提取全局上下文，具有自下而上与自上而下路径，task head 读取缓存特征。可参考多时间尺度记忆。 | 作者实现覆盖分割、检测、深度，包含 Event-only 与额外 RGB 深度版本；不能统一写成全方法均需 RGB。尚未作为本库未来预测方法复现。 | [论文](https://arxiv.org/abs/2305.17852) · [作者项目](https://hamarh.github.io/hmnet/) · [官方 GitHub](https://github.com/hamarh/HMNet_pth) |
| **RED，NeurIPS 2020** | recurrent Event detector，将事件序列压缩到内部记忆，并用 temporal consistency loss 约束训练；适合作为较早的 ConvLSTM 检测与时间一致性参考。 | Event-only 检测；本机主要经 RVT/FLEET 文献引用出现。本轮未核实独立的官方 RED 模型 GitHub，右列工具是数据评测工具，不冒充 RED 模型实现。 | [论文](https://arxiv.org/abs/2009.13436) · [官方数据/评测 GitHub](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox) |
| **E2VID，2019** | recurrent network 将事件窗口恢复为强度图像，hidden state 随历史更新，再复用图像视觉模型。适合“先重建再提取 VFM 特征”的对照。 | 模拟事件和图像监督训练，推理 Event-only；重建优化目标与任务特征目标不同。本机 E-VLA 附录还比较连续更新和按需重建，说明 state 调度会改变实际表现。 | [论文／作者项目](https://rpg.ifi.uzh.ch/E2VID.html) · [官方 GitHub](https://github.com/uzh-rpg/rpg_e2vid) |
| **E-RAFT，3DV 2021** | voxel 编码、跨事件窗口相关性和迭代更新估计 dense flow；可作为事件运动表征／feature matching 与独立运动评价参考。 | 监督训练，推理输入事件；估计已经观测区间的 flow，与未观测未来 flow 不同。官方库可核实 checkpoints 与评估入口，不能据此声称训练代码齐全。 | [论文／作者项目](https://rpg.ifi.uzh.ch/ERAFT.html) · [官方 GitHub](https://github.com/uzh-rpg/E-RAFT) |
| **EV-FlowNet，RSS 2018** | 事件图像式表示进入光流网络，训练以对应灰度图像的 photometric 信息提供监督。可参考“图像用于训练、Event 用于推理”的运动监督范式。 | 训练不需光流 GT；推理 Event-only。其损失与 backbone 目标为 flow，不是 RGB foundation feature 对齐。 | [论文](https://arxiv.org/abs/1802.06898) · [官方 GitHub](https://github.com/daniilidis-group/EV-FlowNet) |
| **HOTS，TPAMI 2017；HATS，CVPR 2018** | HOTS 以 time surfaces 提取时空模式；HATS 将局部 memory 中的 time surfaces 聚合成稳健的分类特征。用来解释近期性、局部统计与时间信息压缩。 | 本机 EST、F³、Event-VLA 引用线索；属于经典特征基线。本轮未核实作者官方 GitHub，不把第三方复现记作官方。 | [HOTS DOI](https://doi.org/10.1109/TPAMI.2016.2574707) · [HATS 论文](https://arxiv.org/abs/1803.07913) |

补充方法的本机定位：HMNet 见 `/root/papers/md/Event/2026_Chen_Scaling-Dense-Event-Stream-Pretraining_Visual-Foundation-Models/auto/2026_Chen_Scaling-Dense-Event-Stream-Pretraining_Visual-Foundation-Models.md` baseline 表；RED 见 `sources/rvt-2023.txt` references [38]；E2VID 见 `/root/papers/md/Event/2026_Zhai_E-VLA_Event-Augmented-Vision-Language-Action/auto/2026_Zhai_E-VLA_Event-Augmented-Vision-Language-Action.md` §8.2；E-RAFT、EV-FlowNet、HATS/HOTS 见 `/root/papers/md/Event/2025_Das_F3_Predictive-Representation-of-Events/auto/2025_Das_F3_Predictive-Representation-of-Events.md` 光流比较与 references。

### DMR：任务相关特征与模态特有噪声分解（CVPR 2024）

- **输入与事件处理：**RGB + Event 的驾驶强化学习。将相邻 RGB 帧之间的事件按 50 ms 窗口转成 5-bin signed voxel；RGB 采样为 20 Hz，策略状态拼接连续三次观察。因此事件在帧间分箱，时序上下文由有限观察堆叠提供。
- **三分支表示：**RGB 与 DVS 两个单模态编码分支提取各自的 noise / task-irrelevant feature；第三分支在输入端拼接 RGB 和事件，提取 co-feature $z_t^c$。这里的 co-feature 是两模态合起来提供的任务相关信息，允许互补，不应收缩解释为两路特征的交集。
- **分解约束：**用 co-feature 加上对应模态的 noise feature 重建各自观察，保证信息完整；contrastive loss 拉开任务相关表示与模态噪声，使用 moving-average targets。训练不依赖像素级任务标注，但重建目标仍来自传感器观察。
- **DeepMDP 与预测：**额外训练动作条件的下一步 latent 与 reward heads，$\widehat z^c_{t+1}=P_\theta(z_t^c,a_t)$、$\widehat r_{t+1}=R_\psi(z_t^c,a_t)$，使 co-feature 保留与状态演化和回报相关的信息；SAC 策略读取 co-feature。论文属于“辅助动态预测帮助策略表征”，未展示靠想象 rollout 优化策略或用 MPC 搜索动作。
- **训练／推理边界：**测试时丢弃两条噪声支路及辅助训练模块，只保留 co-feature encoder 与 policy；该 encoder 仍读取 RGB + Event，所以“只保留中间分支”不等于 Event-only 推理。
- **可借鉴之处：**当直接对齐 RGB 全特征可能把无关外观／噪声也迁移进事件表示时，DMR 提供任务相关性约束、表示分解和重建保真之间的设计参考。其结果来自自定义 CARLA HighBeam / JayWalk、Midnight / Hardrain 场景，不能直接转成真实 DSEC 的未来表示结论；当前无动作项目若复用该思想，需要另定义无动作预测目标。
- **链接：**[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_DMR_Decomposed_Multi-Modality_Representations_for_Frames_and_Events_Fusion_in_CVPR_2024_paper.html) · [官方 GitHub](https://github.com/kyoran/DMR)（本轮可见 `encoder.py`、`transition_model.py`、`agent/`、`env/`、`train_testm.py`）。

本机全文依据：`/root/papers/md/Event/2024_Xu_DMR_Event-Frame-Fusion-Visual-Reinforcement-Learning/auto/2024_Xu_DMR_Event-Frame-Fusion-Visual-Reinforcement-Learning.md` §4.1–4.4、Figure 2、Algorithm 1；本机已有机制复核为 `/root/papers/notes/surveys/event-generative-world-model-survey-2026-10-10.md` 的动作条件模型部分。官方仓库核查日期：2026-11-11。

<a id="alignment"></a>
## 4. RGB / VFM 知识迁移与特征对齐

本节覆盖本机 GEP、ScalE 全文及既有调查中出现的对齐路线。REALM 是此前项目讨论过的补充参考；其方法细节在本轮重读作者原文。其余由 GEP/ScalE related work 引出的工作列为扩展线索，不能把引用过等同于完成复现。GitHub 状态核对日期为 **2026-11-11**。

### 先区分五种对齐对象

| 对齐对象 | 代表方法 | 实际约束什么 | 对 Event 特征的影响 |
|---|---|---|---|
| 全局样本或语义 embedding | ECDP、EventCLIP、EventBind / E-CLIP | 同一对象、图像、文字的语义一致性 | 适合识别和少样本迁移；不能据此推断 dense localization 足够好 |
| 中间特征或最终 patch | Network Grafting、EventSAM、REALM | Event 特征进入已有 RGB 网络或 decoder 可接受的特征空间 | 可直接借用后端；需要处理无事件区与 RGB 稠密外观之间的不对称 |
| 区域与语义结构 | OpenESS、ScalE | 区域内聚合特征，或 token 两两相似性关系 | 从点对点拟合转向对象/场景关系；仍依赖空间配准和可信教师 |
| 输出与任务结构 | ESS、EvDistill、Depth AnyEvent、EventDAM | 分割预测、重建、深度等任务信号 | 迁移目标更明确，通用表示能力需要另外评估 |
| 跨模态或跨时间预测 | GEP、PEPR | 用一段/一种模态的特征预测另一段/另一模态的特征 | GEP 的时序预训练与 PEPR 的同场景跨模态预测不能混写成相同任务 |

### ScalE / ScaleEvent：面向稠密特征的结构蒸馏

**论文**：[Scaling Dense Event-Stream Pretraining from Visual Foundation Models](https://arxiv.org/abs/2603.03969v1)。**GitHub**：[zhiwen-xdu/ScaleEvent](https://github.com/zhiwen-xdu/ScaleEvent)，官方实现、训练入口和权重链接已公开；官方当前 README 标为 CVPR 2026 Highlight。以下具体机制以本地存档 **arXiv v1** 为准。

- **输入与编码**：同步且空间对齐的 Event–RGB 对。事件先做 motion-adaptive sampling，再转为 **3-bin voxel grid**，不是三通道正负计数伪彩图。图像教师与事件学生都使用 DINOv3 初始化的 ViT-S/B/L、patch size 16；图像教师冻结，事件 encoder 全参数更新。
- **为何不只做 patch L1**：事件集中在运动边缘，RGB token 还含对象内部纹理与静态区域。论文观察到高分辨率下直接 patch 或 superpixel 对齐会损害特征的局部区分度，因此先根据事件密度选有效位置，再让学生匹配教师的 token 关系。
- **对齐粒度与损失**：设事件、图像 token 为 $K,Q$，活动 mask 后为 $K^*,Q^*$。除 $\|K^*-Q^*\|_1$ 外，还约束事件内部关系 $K^*K^{*\top}$ 与教师关系 $Q^*Q^{*\top}$ 一致，以及事件到图像关系 $K^*Q^{*\top}$ 与教师关系一致：

  $$
  \mathcal L_{\mathrm{ScalE}}=\mathcal L_{1}(K^*,Q^*)
  +\lambda_{is}\|K^*K^{*\top}-Q^*Q^{*\top}\|_1
  +\lambda_{cs}\|K^*Q^{*\top}-Q^*Q^{*\top}\|_1.
  $$

  这两个结构项约束的是空间 token 间的关系；原文实现相关归一化和 mask 细节应按代码复用，不能把该式解释为任意 Gram loss 都等价。
- **训练与推理模态**：预训练需要配对 RGB，推理事件分支只用 Event；分割、深度、光流任务接各自 decoder，并按 linear probe、few-shot、full fine-tuning 分开评估。论文利用多数据源约 500K 配对样本，不应把收益全部归给单个损失项。
- **最直接的参考价值**：dense patch 对齐、activity mask、跨模态结构保持，以及“蒸馏后是否还保有可定位的细节”的诊断。它为 Event encoder 迁移提供证据；未知未来预测能力、严格历史截止下的收益仍要另外测。

本地来源：`/root/papers/md/Event/2026_Chen_Scaling-Dense-Event-Stream-Pretraining_Visual-Foundation-Models/auto/2026_Chen_Scaling-Dense-Event-Stream-Pretraining_Visual-Foundation-Models.md`，§3、§4、补充材料 Tables 14–18；官方 [CVPR 版本](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Scaling_Dense_Event-Stream_Pretraining_from_Visual_Foundation_Models_CVPR_2026_paper.pdf) 可用于正式引用。

### GEP：基础模型对齐后，再做混合序列自回归预训练

**论文**：[Generative Event Pretraining with Foundation Model Alignment](https://arxiv.org/abs/2603.23032v1)，CVPR 2026 **Findings**。**GitHub**：[uzh-rpg/generative_event_pretraining](https://github.com/uzh-rpg/generative_event_pretraining)，官方两阶段训练、下游训练与 encoder 权重入口已公开。

- **事件输入**：固定时间窗口内生成三通道伪帧，依次为正事件计数、是否有事件的 activity mask、负事件计数。正负通道按联合第 99 百分位截断归一化。窗口内精细时间戳不显式保留；它与 ScalE 的 3-bin temporal voxel 是两种不同输入。
- **Stage 1**：图像与事件 encoder 均从 DINOv2 初始化，图像分支冻结。§3.3 的正式目标是 Event–RGB **cosine alignment + projection-head InfoNCE**，再让事件 encoder 同时读取 RGB，以 preservation loss 维持其原有图像语义空间：

  $$
  \mathcal L_{\mathrm{align}}=
  \lambda_{\cos}\big[1-\cos(E_e(X_e),E_i(X_i))\big]
  +\lambda_{\mathrm{NCE}}\mathcal L_{\mathrm{InfoNCE}}
  +\mu\big[1-\cos(E_e(X_i),E_i(X_i))\big].
  $$

  图像教师输出 stop-gradient。主干特征对齐与投影后的样本级 contrastive discrimination 承担不同角色。
- **Stage 2**：缓存已经对齐的 Event/Image patch tokens，将其交错成混合序列，加入位置和模态编码，用新的 causal Transformer 预测后续 token 序列；回归目标为下一序列切片的特征 MSE。空间特征保留程度对 dense downstream 很关键。
- **梯度边界**：官方 README 要求先生成 `eventToken` / `imageToken` 的 `.pt` 文件，再训练 `src/gpt.py`；其训练函数对输入和目标显式 `clone().detach()`。因此该实现中 **第二阶段的预测损失不更新第一阶段 Event encoder**。这是“先固定表示，再学动态”的机制，不是 end-to-end 未来监督塑形 Event encoder。代码依据：[训练脚本](https://github.com/uzh-rpg/generative_event_pretraining/blob/master/src/gpt.py)、[数据集读取](https://github.com/uzh-rpg/generative_event_pretraining/blob/master/src/dataset.py)。
- **训练与推理模态**：预训练使用 Event、配对图像及混合序列；论文下游感知测试包含 Event 识别、分割和深度。它的 generative pretraining 不等于已经完成当前项目的“只给历史 Event、预测未知未来、用独立 future-task 指标验证”协议。
- **参考价值**：这是 VFM 语义转移和 Event 时间建模组合最直接的近邻。可比较静态对齐、固定 encoder 的动态学习、预测损失回传 encoder 三种训练方式；不能仅以“VFM + Event + future prediction”作为新颖性。

本地来源：`/root/papers/notes/surveys/event-world-action-research/sources/2603.23032.txt`，§3.2–3.4；同论文亦收录于 `md/Event/2026_Cao_Generative-Event-Pretraining_Foundation-Model-Alignment/auto/`。论文引言曾用 MSE 概括 Stage 1，但方法式 (3)–(6) 写的是上述 cosine/InfoNCE/preservation；官方 CLI 另支持损失组合，复现时应注明具体配置。

### REALM：对齐到可直接复用 RGB 后端的特征空间

**论文**：[REALM: An RGB- and Event-Aligned Latent Manifold for Cross-Modal Perception](https://arxiv.org/html/2605.00271v4)，本轮读取 **v4（2026-09-18）**。**作者页**：[STARS / REALM](https://papers.starslab.ca/realm/)。**GitHub**：[utiasSTARS/REALM](https://github.com/utiasSTARS/REALM)，官方代码和模型入口已公开，作者标注 ECCV 2026。

- **表示与 backbone**：5-bin voxel grid 经可训练 convolutional stem 转为 patch tokens，再进入 DUNE 初始化的 ViT。原 backbone 参数冻结，在 attention、projection、feed-forward 层加入 LoRA；主要对齐最终 latent，不要求输入 embedding 本身相等。
- **对齐**：对最终空间 token 使用 MSE、cosine 与 L1 的加权组合。当前版本有两种 mask：事件占用区驱动的 **progressive spatial distillation mask** 随训练逐步膨胀；warm-up 后随机遮掉部分输入 event tokens，迫使表示使用更大范围的上下文。
- **训练与推理**：训练需要同步配准 RGB–Event；部署 Event encoder 时无需 RGB。论文把 RGB teacher 上训练的 linear heads 和冻结 MASt3R 几何 decoder 直接用于 Event 特征，检验特征接口是否真的兼容。
- **参考价值与边界**：这是“对齐后能否直接使用冻结 RGB 后端”的强参考，也适合测试同一个预测器是否能读取 Event 特征。其主要验证是当前深度、分割、匹配/位姿；不同条件下与 RGB teacher 的差距不一致，不能概括成普遍 RGB parity。其固定 voxel 输入也不是原生逐事件更新的长期记忆。

来源范围：本轮核对上述作者论文 §3.3、作者仓库；此前项目讨论中的方法补充，未把它误记为本机已有独立全文归档。

### EventSAM：优先校准对高层输出影响最大的中间 tokens

**论文**：[Segment Any Events via Weighted Adaptation of Pivotal Tokens](https://arxiv.org/html/2312.16222)；正式 CVPR 2024 标题为 *Segment Any Event Streams via Weighted Adaptation of Pivotal Tokens*。**GitHub**：[zhiwen-xdu/EventSAM](https://github.com/zhiwen-xdu/EventSAM)，官方训练、预测、评估与权重入口已公开；旧作者链接 `happychenpipi/EventSAM` 对应同项目。

- **输入与 encoder**：使用 **3-channel time-voxel**，不是图中用于展示的 polarity image。保留 SAM ViT 架构，以原 RGB SAM 为教师；学生主要微调各 block 的末端 MLP，并在训练时把少量 Event embeddings 替换为配对 RGB embeddings。
- **对齐机制**：多层 token L1 蒸馏；由教师 attention 的跨层传播关系估计中间 token 对最终输出的影响，用相应权重加强 pivotal tokens 的对齐。它允许不重要位置存在一定模态差异，目标是更好地匹配供 SAM decoder 读取的末层特征。
- **模态与用途**：训练用配对 RGB–Event，推理用 Event 加原任务所需 prompts。适合借鉴轻量适配、混合 tokens 保持知识、层间重要性加权；验证的是提示式对象分割，不能据此认定时序动态被保留。

本机线索：ScalE §2.2 及相关工作引用；本轮重读 EventSAM §3，并核对官方仓库。

### Network Grafting / NGA：替换前端，保留预训练网络后端

**论文**：[Learning to Exploit Multiple Vision Modalities by Using Grafted Networks](https://arxiv.org/abs/2003.10959)，ECCV 2020；[正式论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610086.pdf)。**GitHub：本轮未核实到作者公开仓库**。

- **输入**：固定事件数窗口，论文用 25,000 events，经时间双线性权重累积成 3 或 10 slices 的 signed voxel volume。
- **网络**：以 COCO 预训练 YOLOv3/Darknet-53 为案例，将原 RGB 前端替换为可训练 Event front end，保留 middle/back-end。新前端输出与原前端相同 shape 的特征。
- **损失**：前端特征 MSE、经过固定 middle network 后的特征 MSE，以及前端/中间特征去均值后的 channel Gram style loss。后两项检验输入特征对后续网络的意义，而不是只拟合一个孤立接口。
- **模态与边界**：训练需要同步图像–Event、无需相应 Event 任务标签，推理只需 Event。最直接的借鉴是固定已训练后端、单独训练传感器前端。Event 检测实验使用 MVSEC 单序列划分及图像检测器伪标测试框，证据范围小于大规模独立人工标注测试；不能把 thermal 分支的增益移作 Event 结论。

本机线索：GEP §2.1 的 Hu 等 grafting 引用；本轮核对正式论文 §3–4。

### PEPR：Event 作为训练期特权信号，部署仍是 RGB

**论文**：[PEPR: Privileged Event-based Predictive Regularization for Domain Generalization](https://arxiv.org/html/2602.04583v2)，**v2（2026-04-17）**，CVPR 2026 Findings；[作者页](https://niccobiondi.github.io/projects/pepr/)。**GitHub**：[miccunifi/PEPR](https://github.com/miccunifi/PEPR)，官方公开仓库。

RGB encoder 产生空间特征，Transformer decoder 以目标位置 query 预测 Event encoder 的局部 latent；损失是 RGB 任务监督加 patch feature MSE。当前在线原文的目标采样混合高活跃和低/零活跃事件区域，并非只采活跃 patch。训练使用 RGB、Event 和任务标签，推理移除 Event encoder 与辅助 predictor，只保留 RGB task model。可借鉴“对齐 readout/预测目标，而不强迫整个状态等同另一模态”的设计；其预测是跨模态监督，不是由历史预测未知未来，模态方向也与 Event-only 部署相反。

本地来源：`/root/papers/notes/surveys/event-rgb-vjepa2-design-2026-10-10/prior-work.md`；本轮按上述在线原文修正目标 patch 采样说明。

### 其他已出现的预训练和稠密对齐线索

下表中 ECDP、EventCLIP 来自已有专门调查；ECDDP、EventBind、ESS、EvDistill、OpenESS、Depth AnyEvent、EventDAM、STP 来自 GEP/ScalE 正文的 related work 或实验基线。本轮核对的是作者方法说明与代码入口，未逐项精读全部补充实验。

| 方法与论文 | Event 表征、处理和对齐机制 | 训练 → 推理模态；可参考之处 | GitHub 与核实状态 |
|---|---|---|---|
| **ECDP**, [Event Camera Data Pre-training, ICCV 2023](https://arxiv.org/abs/2301.01928v3) | Event images 经增强、conditional masking 选有信息的 patches，ViT 编码；event–event embedding projection 与 Event–RGB distribution alignment / contrastive learning。官方代码先提取 CLIP 图像特征，MoCov3 checkpoint 用于初始化 | 配对 Event–RGB → Event；参考事件增强、信息区域采样和全局语义迁移，不宜把分类 embedding 直接当 dense 表示 | [Yan98/Event-Camera-Data-Pre-training](https://github.com/Yan98/Event-Camera-Data-Pre-training)，代码与预训练权重入口 |
| **ECDDP**, [Event Camera Data Dense Pre-training, ECCV 2024](https://arxiv.org/abs/2311.11533) | 从 Event patch 特征挖掘 contextual similarity，按上下文分组后做 context-to-context 自监督，避免大量空 patch 让稠密训练无效；包含 ViT/Swin 路线 | Event-only 预训练 → Event；是与跨模态蒸馏比较的重要原生 Event dense baseline，不能误写为持续依赖 RGB teacher | [Yan98/Event-Camera-Data-Dense-Pre-training](https://github.com/Yan98/Event-Camera-Data-Dense-Pre-training)，训练/分割脚本和样例 checkpoint 已存在；README 仍保留旧 coming-soon 文本 |
| **EventCLIP**, [2023 原文](https://arxiv.org/abs/2306.06354v3) | 事件窗口转 2D grid/图像式表示，调用 CLIP image encoder 与 text classifier；少样本版用轻量特征适配聚合多段 Event 特征 | 零样本使用 Event + 类别文本；少样本版增加 Event 标注训练。参考低成本视觉预训练迁移和 few-shot 对照；主任务是对象识别 | [Wuziyi616/EventCLIP](https://github.com/Wuziyi616/EventCLIP)，官方零样本、少样本、微调及无标签学习实现 |
| **EventBind / E-CLIP**, [EventBind, ECCV 2024](https://arxiv.org/abs/2308.03135) | Event frame 序列经 CLIP 改造的 event encoder，使用 temporal encoding、cross-frame prompts 和 event prompts；HTCA 对齐 Event、Image、Text embeddings | Event/Text，Image 分支可选；推理按识别/检索协议读取 Event 与文本或检索目标。参考保留时间信息的语义迁移；E-CLIP 是同一 arXiv 的早期标题，合并计数 | [jiazhou-garland/EventBind](https://github.com/jiazhou-garland/EventBind)，官方实现；README citation 尚保留 E-CLIP 旧名 |
| **ESS**, [Learning Event-based Semantic Segmentation from Still Images, ECCV 2022](https://rpg.ifi.uzh.ch/docs/ECCV22_Sun.pdf) | 使用 recurrent、motion-invariant event embedding，以重建与语义预测约束把 Event 与 image 表示连接起来；不要求图像–Event 逐像素成对 | 有标签 still images + 无标签 Event → Event 分割；参考无严格配对时的迁移、重建表征接口和 DSEC-Semantic 协议 | [uzh-rpg/ess](https://github.com/uzh-rpg/ess)，官方代码 |
| **EvDistill**, [CVPR 2021](https://arxiv.org/abs/2111.12341) | 多通道 Event 图像输入 student；借 bidirectional reconstruction 缩小模态差距，传递 image teacher 的 feature/task 知识，包含配对与非配对学习设定 | 图像教师、Event 及相应训练设定 → Event 下游预测；参考重建辅助的知识迁移，需分别核对配对/非配对协议 | [addisonwang2013/evdistill](https://github.com/addisonwang2013/evdistill)，公开评估、模型与可视化；不能由仓库存在推断全训练链完整 |
| **OpenESS**, [CVPR 2024](https://arxiv.org/abs/2405.05259) | Event encoder 可处理 voxel / reconstruction；用 SAM 或 SLIC superpixels 聚合区域，做 frame-to-event contrastive distillation 与 text-to-event semantic consistency | Event–Image–Text 预训练 → Event + 类别词汇分割；参考 region-level alignment 与开放词汇 dense readout | [ldkong1205/OpenESS](https://github.com/ldkong1205/OpenESS)，官方预训练、分割与数据配置 |
| **Depth AnyEvent**, [ICCV 2025 作者页](https://bartn8.github.io/depthanyevent/) | 利用与事件空间配准的 RGB，经 VFM 生成 dense proxy depth；适配 Depth Anything v2 为 Event 模型，并提供 recurrent 版本 DepthAnyEvent-R | 配对 RGB–Event + 教师 proxy depth → Event-only depth；参考任务输出蒸馏与 recurrent 深度表征。不是与 EventDAM 同一篇 | [bartn8/depthanyevent](https://github.com/bartn8/depthanyevent)，官方代码 |
| **EventDAM**, [Depth Any Event Stream, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_Depth_Any_Event_Stream_Enhancing_Event-based_Monocular_Depth_Estimation_via_ICCV_2025_paper.html) | 稀疏 Event 与 dense RGB：SFM 按 sparsity metric 混合特征产生辅助深度，SFD 按稀疏度加权特征蒸馏，SCM 约束不同事件稀疏度下的输出一致性 | 配对 RGB–Event、无人工深度标签的蒸馏 → Event-only depth；参考“不同事件密度下表示/输出一致”的约束 | **本轮未核实到作者公开 GitHub**；不能借用 Depth AnyEvent 仓库充当其代码 |
| **STP**, [Efficient Event Camera Data Pretraining with Adaptive Prompt Fusion, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Liang_Efficient_Event_Camera_Data_Pretraining_with_Adaptive_Prompt_Fusion_ICCV_2025_paper.pdf) | TECM 保留时间片与极性，overlap patch embedding + temporal Transformer / local temporal aggregation 形成 Dense Event Stack；输入冻结 image-pretrained ViT，由分类交叉熵训练 STP 与新 patch embedding，微调时联合更新 | 有标签 Event 预训练，使用 RGB 预训练权重 → Event；参考小规模 adapter 保留稀疏时空细节，适用分类、分割、光流；**不是无标签跨模态蒸馏** | [Lqm26/STP](https://github.com/Lqm26/STP)，官方代码 |

### 本轮核对中需要保留的版本更正

1. **ScalE 与 GEP 的三通道输入不同**：前者是时间 voxel，后者是正计数 / 活跃 mask / 负计数；EventSAM 同样使用 time-voxel，论文 polarity 可视化不等于模型输入。
2. **GEP 两阶段并不端到端**：官方第二阶段读缓存 token 并 detach；Stage 1 损失以 §3.3 正式公式和配置为准，不能只抄引言中的 MSE 概括。
3. **REALM mask 是版本相关细节**：当前在线 §3.3 写 progressive dilation 加 input patch masking；旧的“只对有事件区域蒸馏”不足以描述当前方案。
4. **PEPR 采样包含低/零活跃区**：应与旧本地调查的简写区别；其部署模态是 RGB-only。
5. **ECDP / ECDDP 不同**：ECDP 使用 Event–RGB 监督，ECDDP 强调 Event-only dense self-supervision；MoCov3 初始化与 CLIP 特征监督不可混同。
6. **EventBind / E-CLIP 合并，Depth AnyEvent / EventDAM 分开**：前者是同一项目演进；后者作者、论文、方法和仓库均不同。

7. **STP 使用有标签分类预训练**：正式论文 §3.3/Fig.3 明确用 cross-entropy，而非把它归入无标签对齐或自监督。

<a id="prediction"></a>
## 5. 预测表示、未来生成与应用模块

### 5.1 预测目标如何塑造 Event 特征

#### F³ / Fast Feature Field：直接从稀疏事件学习可预测特征

- **论文**：[Fast Feature Field (F³): A Predictive Representation of Events](https://arxiv.org/abs/2509.25146)，本机为 v2（2026-07 修订；首发 2025）。**GitHub**：[grasp-lyrl/fast-feature-fields](https://github.com/grasp-lyrl/fast-feature-fields)，已公开 encoder、训练配置、预训练权重及光流/深度/分割下游入口。
- **输入与提取**：对过去窗口中的事件坐标 `(t,x,y)` 使用 multi-resolution hash encoding；只查询出现事件的位置。每个像素在时间维做 sum pooling，再经空间 CNN 聚合邻域，输出稠密多通道特征图：

  $$h_t(u)=\rho\!\left(\sum_{s\in\mathcal E_{[t-\Delta t,t)}(u)}\phi(s,u)\right).$$

  这是 Deep Sets 式聚合：对输入枚举顺序不敏感，但单个事件的时间坐标已进入 `φ`，因此并非丢弃时序。论文默认历史窗口 20 ms、32 通道输出；hash 查表利用稀疏性，后续空间 CNN 仍处理稠密特征图。
- **处理与监督**：线性预测头从 `h_t` 预测下一窗口每个时空位置的事件发生概率，使用 class-weighted focal loss 应对极稀疏正样本和噪声；随后把特征用于语义分割、深度、跨时间/双目匹配及光流。核心训练目标是原生事件预测，没有 RGB/VFM 教师。
- **可借鉴**：强 Event-only predictive encoder；可以分别比较手工 voxel、纯重建预训练、未来事件预训练与 RGB 特征蒸馏。它最直接回答“未来预测能否使 event 表征同时保留结构与运动”。论文中的主要下游验证是感知任务；若用作未来检测 encoder，仍需单独测量未来任务效果。
- **本机出处**：全文 §2.3–2.6（`/root/papers/md/Event/2025_Das_F3_Predictive-Representation-of-Events/auto/2025_Das_F3_Predictive-Representation-of-Events.md`）。

#### E-Motion：将视频扩散先验迁移到未来事件帧

- **论文**：[E-Motion: Future Motion Simulation via Event Sequence Diffusion](https://arxiv.org/abs/2410.08649)，NeurIPS 2024。**GitHub**：[p4r4mount/E-Motion](https://github.com/p4r4mount/E-Motion)，已公开训练、推理、预处理及 checkpoint 入口。
- **表示与处理**：把事件压成 `B=3` 的 voxel/frame-like 表示，以适配 Stable Video Diffusion 的图像接口；训练利用事件与对应图像，将 RGB 视频生成先验迁移到事件域。随机改变事件窗口范围，使模型适应运动时间尺度。
- **条件对齐**：推理输入多个高时间分辨率的历史事件帧；在扩散过程中，把加噪后的已知 event prompt 放回序列前部，给后续未知帧提供运动条件。另用 PPO 优化基于生成质量的 reward；此处 RL 调整生成器，不是机器人动作控制。
- **可借鉴与边界**：可参考 event-to-video-model 的通道/latent 接口、多历史 prompt 和生成监督。它预测未知未来，但输出是离散事件帧/体素，并非完整的连续时间四元组流；也没有候选控制动作输入。
- **本机出处**：归档全文 §4（`/root/papers/notes/surveys/event-world-action-research/sources/2410.08649.txt`）、生成模型综述（`/root/papers/notes/surveys/event-generative-world-model-survey-2026-10-10.md`）。

#### UniDynamics：Event–RGB latent 对齐与多任务未来生成

- **论文**：[UniDynamics: Event-RGB Fusion for Unified Future 4D Dynamic Scene Generation](https://arxiv.org/abs/2610.03473)，2026；作者页面标注 ECCV 2026。**GitHub**：[KK-xi/Unidynamics](https://github.com/KK-xi/Unidynamics)，本次仍仅见 README，不能视作可复现实现。
- **输入/提取**：当前 RGB `I_T` 与截至当前的历史事件 `E_(T-1→T)`；100 ms 事件窗口以时间线性插值构成 15-bin signed voxel，经 2D ResBlocks 编码为 `F_e`。
- **对齐/融合**：ELE（Event Latent Enhancement）先让 event features 与 RGB VAE latent 双向 cross-attention，卷积融合后，再以 RGB latent 为 key/value 做 cross-attention，得到与视频扩散 latent 兼容的 `Z_E`。该 event latent 含实际 RGB 外观信息，不能把它视为 Event-only encoder 的输出。
- **时间注入**：把 `Z_E` 当作序列中的一个历史状态 latent，与未来 noisy latents 一起加噪；当前 RGB latent 则作为各时间步的参考条件。事件提供初始运动线索，避免把短时变化强制重复为所有未来帧的硬条件。
- **联合处理**：PDS（Perceptual Dynamics Space）在 U-Net 多尺度中分出 depth/flow adapters；深度分支用 2D、flow 分支用 3D 卷积，门控交换信息，再分别反馈到 spatial/temporal 层。最终共同生成未来 RGB、depth、flow。
- **可借鉴与边界**：参考 latent 尺寸/语义兼容、运动和外观的不同注入位置，以及不同任务特征的门控交换。它属于当前 RGB + 历史 Event 的被动未来生成，推理需要 RGB，没有动作条件；作为 Event-only 项目的机制参考，需移除该 RGB 依赖后重新验证。
- **本机出处**：全文 §3.2–3.5（`/root/papers/md/Event/2026_Liu_UniDynamics_Event-RGB-Future-4D-Scene-Generation/auto/2026_Liu_UniDynamics_Event-RGB-Future-4D-Scene-Generation.md`）。

#### ECHO：用视觉变化监督 Event encoder，并保留运动方向

- **论文**：[ECHO](https://arxiv.org/abs/2609.34893)，2026 预印本。**GitHub**：未确认公开算法仓库；[作者项目页](https://echo-wam.github.io/)提供方法、论文及演示，网页地址不作为训练代码。
- **表示与 encoder**：截至当前的事件窗口分成 6 个等时 signed bins，计数缩放并裁剪到 `[-1,1]`；convolution–attention encoder 结合 patch embedding、block/grid attention、卷积/下采样及 full attention，得到 64 个空间 event tokens。
- **教师对齐的关键**：冻结 DINOv3，聚合/压缩多层 RGB 特征到与 event tokens 对应的空间网格；事件 latent 不直接拟合静态 RGB，而是解释两帧教师特征之间的变化。warp head 预测 feature displacement；synthesis head 补充新显露内容；gate 混合两分支预测后一帧教师特征：

  $$\hat u_{t+\Delta}=\alpha\odot\operatorname{Warp}(u_t,g_w(E))+(1-\alpha)\odot g_s(E,u_t).$$

  重建采用按教师变化幅度加权的 SmoothL1。duration-jittered 事件窗口作为正样本，时间反转并翻转极性的窗口 `-Flip(V)` 作为反向样本，使用 motion-direction contrast；再加 variance floor 防止塌缩。
- **时间边界**：上述 encoder 预训练使用 `t→t+Δ` 区间内已经观测的事件解释同一区间的 RGB 特征变化，属于跨模态变化建模。下游策略中的独立 foresight queries 才预测动作时域里的未来 event latent，并用 frozen event encoder 的未来输出作 stop-gradient target。
- **记忆与融合**：当前 event tokens、RGB/语言、访问过的夹爪空间 cell tokens、foresight queries 进入策略 prefix。所谓 off-camera memory 保留访问 cell 位置，且每个 cell token 加入当前事件均值；不是存储每个历史地点的完整历史外观。foresight 位于 action expert 之前，公开接口尚未展示任意候选动作输入后的未来 rollout。
- **可借鉴**：直接相关的是“对齐变化而非只对齐外观”、warp+synthesis 分工、时间反转负样本与 future-event 辅助监督。推理仍用腕部 RGB、语言、本体状态及事件，不能当作已经验证的 Event-only 无动作预测系统。
- **本机出处**：全文 §III-B、IV-B–D（`/root/papers/md/Event/2026_Wang_ECHO_Event-Augmented-Context-for-Wrist-Only-Manipulation/auto/2026_Wang_ECHO_Event-Augmented-Context-for-Wrist-Only-Manipulation.md`）。

### 5.2 原始事件 token、VLM/VLA 接口和任务驱动融合

#### FLEET：原始事件采样、时间门控与固定 latent bottleneck

- **论文**：[FLEET](https://arxiv.org/abs/2608.16523)，2026 预印本。**GitHub**：[tgottwald/FLEET](https://github.com/tgottwald/FLEET)，已公开 agent、Event-CarEnv、训练/评估脚本。
- **提取与压缩**：将一个观测窗口内按时间排列的事件，沿事件索引划成 `M` 段，每段随机取一个；不足时 padding 并 mask。空间坐标和时间分别归一化，再做 random Fourier features（RFF），由时间 embedding 的 sigmoid gate 调制空间 embedding：

  $$z_i=\operatorname{RFF}(x_i,y_i)\odot\sigma(W_t\operatorname{RFF}(t_i)+b_t).$$

  `N` 个可学习 queries 对事件 tokens cross-attention，压成固定数 latent tokens，经 MLP 与平均池化送入 PPO。
- **训练**：不需要 RGB 或重建预训练；actor/critic 共享提取器，但 actor 输入 stop-gradient，encoder 实际由 critic value loss 更新。采样和 cross-attention 仍依赖读入/选中的事件数；只有压缩后的 backbone 规模固定。
- **可借鉴与边界**：适合比较 raw-event set→固定 token 与 voxel→dense map 的取舍，尤其是 event-rate 变化和 token 预算。控制奖励训练出的全局特征不自动等同于适合像素级空间预测的表示，需要保留或恢复空间布局。
- **本机出处**：全文 §4（`/root/papers/md/Event/2026_Gottwald_FLEET_Event-Token-Reinforcement-Learning/auto/2026_Gottwald_FLEET_Event-Token-Reinforcement-Learning.md`）。

#### EventDrive：多时间分辨率专家 + Event Q-Former

- **论文**：[EventDrive](https://arxiv.org/abs/2606.18242)，CVPR 2026。**GitHub**：未确认完整模型实现仓库；论文给出的泛化 `github/EventDrive` 入口本次返回 404。已公开资源为 [Hugging Face 数据、标注与评估脚本](https://huggingface.co/datasets/dylanorange/EventDrive/tree/main)。
- **表示/encoder**：极性分离 voxel 张量 `2×B×H×W`，用 `B={20,50,100}` 构造多时间分辨率分支；RVT/MaxViT–RNN 类 backbone 提取事件特征。这里原文公式改变的是同一窗口中的时间 bin 数，不宜直接改写成三个不同长度的历史窗口。
- **融合/对齐**：多专家输出参与门控，选择相应时间分辨率；Event Q-Former 用可学习 queries 从事件 feature map 抽取固定数量 motion tokens，线性投影至 Qwen2.5-VL hidden space，再与图像和文字 tokens 合并。先冻结 LLM/帧编码器，只训练事件路径进行 event–language caption 对齐，再进行多模态 instruction tuning。
- **可借鉴**：动态时间分辨率选择、query bottleneck 与分阶段对齐；对齐目标来自语言任务而非固定 RGB patch 的逐位置距离。预测/规划任务还含帧、语言及特定历史轨迹/速度信息，现有轨迹结果为离线评估。
- **本机出处**：全文 §4、附录 B.3（`/root/papers/md/Event/2026_Lu_EventDrive_Event-Cameras-for-Vision-Language-Driving-Intelligence/auto/2026_Lu_EventDrive_Event-Cameras-for-Vision-Language-Driving-Intelligence.md`）。

#### Event-VLA：PREI 多时间线索与动作侧门控路由

- **论文**：[Event-VLA: Action-Conditioned Event Fusion](https://arxiv.org/abs/2606.29384)，2026 预印本。**GitHub**：本次未确认作者发布。检索中的另一个 `EventVLA: Event-Driven Visual Evidence Memory` 使用语义事件/keyframe memory，是另一篇工作，不可代填为本论文代码。
- **表示**：PREI（Physical Residual Event Integration）由事件历史形成三通道图：指数时间衰减活动对应 instantaneous cue；活动与局部平滑背景的比值对应 salient cue；累计次数对应 persistent cue。此处活动累积不区分极性，三个通道表达不同统计含义，不是 RGB 或单纯三个时间 bin。
- **提取/对齐**：事件 encoder 产生全局+空间 tokens，用 frozen visual teacher 做 feature distillation，训练对包含 N-ImageNet 与 v2e 生成的 LIBERO pairs。
- **融合**：预训练 VLA 先处理 RGB、语言、本体状态及 learnable queries；这些 queries 再通过 gated cross-attention 选择事件 tokens，并路由到 action slots。事件内容主要注入动作路径，不直接改变全局语义 token 序列。
- **未来目标**：附录事件头显式以当前事件特征、event query、预测动作块为条件预测未来 PREI；这是训练辅助目标，测试策略并非靠该头枚举候选动作。
- **可借鉴**：不同衰减尺度的活动统计、教师监督后的事件侧 token，以及“按下游 query 读取事件”的接口。对无动作场景预测，action-query 需要替换为 scene/spatial/future queries；这种替换属于待验证设计。
- **本机出处**：全文 §3.2–3.3、附录 B.2/B.4（`/root/papers/md/Event/2026_Liu_Event-VLA_Action-Conditioned-Event-Fusion/auto/2026_Liu_Event-VLA_Action-Conditioned-Event-Fusion.md`）。

#### E-VLA：recent-count 窗口与冻结视觉主干的分层 adapter

- **论文**：[E-VLA](https://arxiv.org/abs/2604.04834)，ECCV 2026（作者声明）。**GitHub**：[JJayzee/E-VLA](https://github.com/JJayzee/E-VLA)；本次已确认 [event_adapter 分支文档](https://github.com/JJayzee/E-VLA/blob/event_adapter/docs/event_adapter.md)给出采集、event dataset、训练与机器人部署入口，另有 image_only 分支及 [MultiLight 数据](https://huggingface.co/datasets/JJayzee/E-VLA-MultiLight)。此前本机“仅占位/待发布”状态已过时。
- **时间对齐**：以当前图像曝光结束时刻为 cutoff，选取最近 `N` 个事件；高速时窗口自动变短，低速时变长。窗口的物理时长因此随事件率变化，若要用于固定时间预测，需额外提供时长信息。
- **两条融合路径**：overlay 直接在 RGB 像素上叠加最近事件的极性颜色；hierarchical adapter 则累计不区分极性的 event-count map、归一化，并利用彩色 DAVIS 的 Bayer 格式 demosaic 成三通道输入。
- **adapter 设计**：与图像分支共享 16×16 patch embedding，以 4 个轻量 Transformer blocks 编码事件，在冻结 SigLIP 的第 3/6/9/12 层通过 `MLP(Concat(image,event))` 渐进融合；保持视觉 token 数量，之后接 SmolVLA。训练先适配事件模块，再联合动作头/projection，并随机 dropout 图像路径促使模型使用事件。
- **可借鉴与边界**：特征多层注入、保持 token 数量及 recent-count vs fixed-duration 对照。彩色事件传感器的 demosaicing 不能直接套到单色 DSEC；方法推理需要 Event+RGB，并非事件学生独立替代 RGB 教师。
- **本机出处**：全文 §3.3–3.4、§5.1（`/root/papers/md/Event/2026_Zhai_E-VLA_Event-Augmented-Vision-Language-Action/auto/2026_Zhai_E-VLA_Event-Augmented-Vision-Language-Action.md`）。

#### DRFuser：双 ResNet 的多尺度中层融合

- **论文**：[Multimodal Fusion for Sensorimotor Control in Steering Angle Prediction](https://doi.org/10.1016/j.engappai.2023.107087)，2023。**GitHub**：[munirfarzeen/DRFuser](https://github.com/munirfarzeen/DRFuser)，已公开同步预处理、模型、训练与测试；自采数据需申请。
- **表示/提取**：自采数据按固定 100,000 个事件生成 event image（该采集条件下约 50 ms），与 RGB、控制信号同步；两个 ResNet 分支分别处理事件图和帧。
- **融合**：多个尺度分别做 self-attention，利用位置关系提取长距离特征，再将两模态输出逐元素融合；最后卷积和 MLP 解码转向。它属于多尺度 feature fusion，不是仅对两个全局向量做一次拼接。
- **可借鉴**：适合作为简单、可复现的中层融合 baseline，比较 early/middle/late fusion 与固定时间/固定数量窗口。已有结果是离线转向回归，不能等同于未来场景预测。
- **本机出处**：全文 §3.2.3、4.1.2（`/root/papers/md/Event/2023_Munir_DRFuser_Event-RGB-Steering-Prediction/auto/2023_Munir_DRFuser_Event-RGB-Steering-Prediction.md`）。

#### eNavi：两通道直方图与两个全局 token 的后融合

- **论文**：[eNavi](https://arxiv.org/abs/2603.14397)，2026 预印本。**GitHub**：[eventbasedvision/eNavi](https://github.com/eventbasedvision/eNavi)，已公开模型定义、12 个权重和日志；完整训练/机器人执行流程尚未确认。
- **表示/对齐**：RGB 时间戳之间的事件按 ON/OFF 分别累积成两通道 histogram；同步成图像、事件、由里程计恢复的速度标签。
- **处理/融合**：冻结 RGB MobileNetV3-Small，训练 Event MobileNetV3-Small；两个全局 feature vectors 形成两个 tokens，经过 Transformer encoder，再由 MLP 输出线/角速度。它的注意力融合粒度是全局 modality tokens，不能直接当作 dense patch correspondence。
- **可借鉴**：轻量双编码器及冻结 RGB 路径的最小融合对照；适合检验事件是否带来额外信息。论文主要证据为真实数据离线速度误差。
- **本机出处**：全文 §III-C、IV-A（`/root/papers/md/Event/2026_Ramesh_eNavi_Event-Based-Imitation-Navigation/auto/2026_Ramesh_eNavi_Event-Based-Imitation-Navigation.md`）。

### 5.3 本机其他驾驶、导航、伺服与 TTC 参考

| 工作 | Event 表示、处理/对齐与可借鉴点 | 输入/证据边界 | GitHub 与发布范围 |
|---|---|---|---|
| [DDD17](https://arxiv.org/abs/1711.01458)，2017 | 同步记录 DVS、APS 灰度、车辆 CAN/GPS；提供事件分窗与驾驶监督的早期数据基础 | 数据集；APS 不是彩色 RGB | [SensorsINI/ddd20-utils](https://github.com/SensorsINI/ddd20-utils)：后续共用工具；旧 ddd17-utils 地址在原调研中失效 |
| [Event Steering](https://arxiv.org/abs/1804.01310)，CVPR 2018 | 正/负事件分别累计为双通道 histogram，避免相反极性相互抵消；ResNet18/50 + global average pooling + MLP 预测转向；比较积累时长 | Event-only 转向与 APS 对照；50 ms 是该任务的经验选择 | [uzh-rpg/cvpr18_event_steering_angle](https://github.com/uzh-rpg/cvpr18_event_steering_angle)：预处理、训练、评估 |
| [DDD20](https://arxiv.org/abs/2005.08605)，ITSC 2020 | 50 ms signed ON/OFF count histogram，按 3σ clipping 并归一化；APS 低帧率时复制帧以匹配输入，研究事件/灰度融合 | 离线转向；帧复制不创造新的观测信息 | [ddd20-itsc20](https://github.com/SensorsINI/ddd20-itsc20)：模型；[ddd20-utils](https://github.com/SensorsINI/ddd20-utils)：数据读取转换 |
| [Towards Closing the Domain Gap with Event Cameras](https://arxiv.org/abs/2512.16178)，2025 | 50 ms ON/OFF 事件图对齐 APS；控制日/夜训练比例，比较跨光照行为克隆 | 可参考域偏移实验；按单条记录前/后切分与跨序列泛化是不同协议 | 未确认官方 GitHub |
| [Energy-Aware IL](https://arxiv.org/abs/2603.28008)，2026 | 事件直方图 + 双 ResNet；ECFM 根据通道均值/方差与像素偏离量形成 energy 权重，模态内增强并生成交叉权重，concat 后 1×1 conv 融合；decoder 预测转向及方差 | 轻量统计注意力参考；energy 是函数值，非功耗；离线 Event+Frame | 未确认官方 GitHub；论文仍声明录用后发布 |
| [RTRRL Driving](https://arxiv.org/abs/2602.02236)，2026 | 100 Hz、128×64×2 event frames；CNN/autoencoder 编码后接 LrcSSM，先 BC，再每时间步在线 recurrent RL 更新 | 可参考记忆和在线适应；事件帧丢弃子帧时间；真实证据是小车循线 | 未确认论文对应官方 GitHub |
| [Human-Robot Navigation](https://arxiv.org/abs/2506.10790)，2025 | 10 ms polarity-separated SAE/time surfaces → YOLOv5s 行人检测；检测框与 depth、LiDAR、速度组成低维状态；IL 初始化后 DDPG 控制 | 检测—低维状态—控制的模块化参考；非纯 Event；动作以 0.1 s 更新 | 未确认官方 GitHub |
| [SEBVS](https://arxiv.org/abs/2508.17643)，2025 | ROS2/Gazebo 经 v2e 生成事件；ON/OFF count maps 与 RGB 合成 5 通道输入，16×16 patchify 后轻量 Transformer 和动作头 | early fusion 基线与仿真管线；生成事件仍受源视频/渲染过程约束 | [eventbasedvision/SEBVS](https://github.com/eventbasedvision/SEBVS)：已见 v2e ROS2 节点与数据入口，未确认完整策略训练 |
| [Neuromorphic Eye-in-Hand](https://arxiv.org/abs/2004.07398)，IEEE Access 2021 | 从 SAE 提取事件角点，维护活动特征/锁定目标与虚拟质心，结合事件反馈切换探索、对齐和抓取控制 | 可参考稀疏特征跟踪与主动产生可观测变化；不是 learned semantic encoder | 未确认官方 GitHub |
| [Real-Time Grasping](https://arxiv.org/abs/2107.07200)，2022 | 已知物体：多视角事件定位、点云聚类和模型配准；未知物体：时空 mean-shift（MEMS）、eHarris 角点、目标质心与伺服 | 提供 `(x,y,t)` 时空聚类与对象级读出；模型式分支依赖形状先验 | 未确认官方 GitHub |
| [Event-Aided TTC](https://arxiv.org/abs/2407.07324)，ECCV 2024 | 以事件 normal flow 几何约束初始化 TTC，再用事件时空配准细化；可作为运动特征和低维风险的几何 baseline | 已观测运动 → TTC 估计；不是多步世界生成或自动制动策略 | [NAIL-HNU/event_aided_ttc](https://github.com/NAIL-HNU/event_aided_ttc)：MATLAB 实现、FCWD 等数据入口 |
| [EvTTC](https://arxiv.org/abs/2412.05053)，RA-L 2025 | 双焦距 Event/Frame 与 LiDAR、GNSS/INS 同步/标定，构造 TTC 真值；可参考多传感器对齐与风险评价 | 主要是数据资源；算法发布应与数据网页分开 | [NAIL-HNU/EvTTC](https://github.com/NAIL-HNU/EvTTC)：数据/标定资源，TTC 算法另见 Event-Aided TTC |
| [eAP / Garl-TTC](https://arxiv.org/abs/2603.16303)，2026 | RGB 与 event voxels 编码，学习物体可见高度比，用几何公式推 TTC；训练额外用 SAM 前景 mask 蒸馏，使特征保留轮廓。RGB 映射到 event distorted plane 使用 infinite-depth warp，仍存在视差残差 | 训练时 foreground teacher，推理 Event+RGB；轮廓对齐及 geometry-aware target 对 feature 学习直接相关 | [NAIL-HNU/Garl-TTC](https://github.com/NAIL-HNU/Garl-TTC)：训练/推理/评估/转换；[eAP_dataset](https://github.com/NAIL-HNU/eAP_dataset)：数据网站 |
| [EV-TTC / T²CEF](https://doi.org/10.1109/LRA.2025.3565150)，RA-L 2025 | 从 M3ED 的深度与姿态构造 dense TTC 监督，学习事件到像素级风险；可作为 event representation 的稠密任务评价 | 与上行 EvTTC 是不同工作；本机原调研主要核验仓库/生成流程，未全文精读 | [anthonytec2/EV-TTC](https://github.com/anthonytec2/EV-TTC)：ROS2、训练及 T²CEF 生成 |

本表前 13 项均对应 `/root/papers/md/Event/` 的已归档论文；逐项本地全文链接见 原始文献矩阵（`/root/papers/notes/surveys/event-research/paper_matrix.md`）。其中 eAP/Garl-TTC 的 SAM 教师与几何高度比设计，本次另核对 全文 §VI（`/root/papers/md/Event/2026_Li_eAP_Event-Enhanced-Autonomous-Perception-and-TTC/auto/2026_Li_eAP_Event-Enhanced-Autonomous-Perception-and-TTC.md`）。

### 5.4 已有 world-action 调研中的补充方法

| 工作 | Event 特征与处理机制 | 可借鉴点和边界 | GitHub / 本机出处 |
|---|---|---|---|
| [Motion-aware Event Suppression](https://arxiv.org/abs/2602.23204)，RSS 2026 | 两个时间 bin 的 event stack → ConvGRU encoder；mask decoder 分离 independently moving objects（IMO）；另一分支通过 Attention-based Time Conditioning 将预测时距作为 query，调制空间特征以预测未来 flow，再把 mask warp 到未来以筛选事件 | 自车/独立物体运动分离、按 future horizon 条件化、动态 token pruning 可直接用于事件处理；抑制哪类事件取决于下游，不能固定认为背景或动态事件无用 | [uzh-rpg/event_suppression](https://github.com/uzh-rpg/event_suppression)，有实现；本机全文（`/root/papers/notes/surveys/event-world-action-research/sources/2602.23204.txt`） |
| [Approximate Imitation Learning for Event-based Quadrotor Flight](https://arxiv.org/abs/2603.07578)，2026 | 离线成对事件与状态，使用 teacher actions 学 event student/task latent；state-based approximate student 与事件表示 feature alignment，共用 action decoder；真实 M3ED 辅助距离/深度学习缩小 sim-to-real gap。在线策略改进使用廉价状态替代渲染事件，再把共享动作头接回事件 encoder | 可参考“对齐 task latent 而非 appearance latent”及昂贵模态与廉价代理训练的解耦；真实飞行仍依赖外部位姿信息。当前 arXiv 已有 v3，本文机制依据本机归档 v2 | 未确认公开官方 GitHub；本机全文（`/root/papers/notes/surveys/event-world-action-research/sources/2603.07578.txt`） |


<a id="generation"></a>
## 6. 事件生成、重建与数据工具

这些工作分别提供事件 latent、视觉 decoder、去噪和合成数据方案。它们是否适合作为预测模块，取决于输入的事件时间范围：输入目标区间事件的重建方法，可以用于表征与解码设计，但其重建结果不能直接作为仅凭历史预测未来的证据。

### 6.1 事件 latent、补全与生成

| 工作／论文 | Event 的提取、处理或生成机制 | 对特征研究的参考价值 | GitHub 与核查状态 |
|---|---|---|---|
| **eVAE / Representation Learning for Event-based Visuomotor Policies**（2021）[论文](https://arxiv.org/abs/2103.00806) | 直接编码异步时空事件为 VAE latent，以事件重建约束表示，再用于视觉运动策略 | 事件自监督表示到下游策略的早期接口；重建 latent 本身未定义状态转移 | [microsoft/event-vae-rl](https://github.com/microsoft/event-vae-rl)，有 `event_vae`、`event_rl`，仓库已归档 |
| **EventGAN**（ICCP 2021）[论文](https://arxiv.org/abs/1912.01584) | 两端图像条件生成时空事件体素，以光流／重建的一致性约束生成器 | 利用现有图像标注合成事件训练样本；需要区间两端图像 | [alexzzhu/EventGAN](https://github.com/alexzzhu/EventGAN)，官方实现 |
| **V2CE**（ICRA 2024）[论文](https://arxiv.org/abs/2309.08891) | 视频先转事件体素，再以 local dynamic-aware timestamp inference 恢复连续时间事件 | 比单纯输出累积图更接近原始流；适合配对数据构造和时间表示研究 | [ucsd-hdsi-dvs/V2CE-Toolbox](https://github.com/ucsd-hdsi-dvs/V2CE-Toolbox)，训练、转换及权重入口可见 |
| **EDR**（Scientific Reports 2024）[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10958031/) | 把 $(x,y,t)$ 当作点云、极性作为特征；PointNet++ 风格网络执行 diffusion 粗补全与 refinement，使用适配时空尺度的邻域 | 原生事件集合的编码、条件特征传递和补全目标；实验是同一区间稀疏到稠密补全 | **未确认官方 GitHub**；保留论文入口 |
| **Text-to-Events**（2024）[论文](https://arxiv.org/html/2406.03439v1) | 活跃 patch 过滤、固定事件数切片、正负分离时间分箱；逐级训练 sparse AE，再做文本条件 latent diffusion | AE 的全零坍塌、极性／bin 失衡与 occupancy loss 设计；生成对象首先是分箱表示 | **未确认官方 GitHub** |
| **S2D-LFE**（CVPR 2025）[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_S2D-LFE_Sparse-to-Dense_Light_Field_Event_Generation_CVPR_2025_paper.pdf) | 事件 VAE 与扩散模型，以视角条件完成稀疏到稠密光场事件生成 | 共享事件 latent 与跨视角结构；这里预测缺失视角，未定义未来动作后果 | [Yutong2022/S2D-LFE](https://github.com/Yutong2022/S2D-LFE)，模型、采集与处理目录可见 |
| **ControlEvents**（WACV 2026）[论文](https://arxiv.org/abs/2509.22864) | 微调 Stable Diffusion / ControlNet，以类别文本、2D 骨架或3D姿态生成事件累积图 | 低数据条件的基础模型迁移和带标注合成；累积图缺少完整异步时间结构 | [项目页](https://yuxuan-xue.com/controlevents/)链接到 [YuxuanSnow/controlevents](https://github.com/YuxuanSnow/controlevents)，本轮访问 **404**，不能标作可用实现 |
| **EDFilter / Event Stream Filtering via Probability Flux Estimation**（2025预印本／2026正式稿）[论文](https://arxiv.org/html/2504.07503) | 对亮度越阈值的 probability flux 做时序核密度估计和运动相关的空间平滑，再重采样事件 | 利用极性与事件间隔，而非仅删掉噪声点；是连续密度滤波，不是学习语义的扩散网络 | **未确认官方 GitHub** |
| **EGDeblurring**（CVPR 2025）[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_Diffusion-based_Event_Generation_for_High-Quality_Image_Deblurring_CVPR_2025_paper.pdf) | 训练中利用事件形成 motion prior；推理由模糊图像生成该先验特征，经双注意力通道融合用于去模糊 | 可借鉴“保留任务所需的事件信息”与辅助模态训练；生成的是特征，不是 raw events | [XinanXie/EGDeblurring](https://github.com/XinanXie/EGDeblurring)，三阶段训练与测试脚本可见 |

### 6.2 事件条件的图像／视频恢复

| 工作／论文 | 编码、对齐与时间处理 | 主要输入边界 | GitHub 与核查状态 |
|---|---|---|---|
| **Event-based HDR Image and Very High Frame Rate Video Generation using Conditional GANs**（CVPR 2019）[论文](https://arxiv.org/abs/1811.08230) | 从事件表示学习图像生成，提供较早的 event-to-image adversarial decoder 路线 | 输出强度／视频而不是事件；观测事件对应待重建内容 | **未确认官方 GitHub** |
| **TimeRewind → Repurposing Pre-trained Video Diffusion Models for Event-based Video Interpolation**（2024→CVPR 2025）[早期论文](https://arxiv.org/abs/2403.13800) | 将事件条件通过 adapter 注入预训练视频扩散；早期版本根据锚图像与已记录事件恢复过去 | 有目标区间事件；后续论文扩展到插帧。按同一研究脉络理解 | [codingrex/TimeRewind](https://github.com/codingrex/TimeRewind)，有模型与推理 demo；README 指向后续版本 |
| **EvDiff**（2025预印本；仓库标 ECCV 2026）[论文](https://arxiv.org/abs/2511.17492) | 5-bin voxel → recurrent EvEncoder → SD3 latent；一步扩散与时序约束产生彩色视频 | 输入帧间事件，是重建。公共训练代码替换了原实验的内部 teacher，作者说明替换训练未验证复现原结果 | [LiWeitu/EvDiff](https://github.com/LiWeitu/EvDiff)，训练、推理与数据转换代码可见 |
| **EvFRA（早期名 DESSERT）**（2025／2026修订）[论文](https://arxiv.org/html/2512.17323v3) | 锚帧提供外观，事件提供帧间变化；学习 residual synthesis / feature adaptation | 目标区间事件已知；新旧标题不要计成两篇独立方法 | [jiyun-kong/EvFRA](https://github.com/jiyun-kong/EvFRA)，模型与训练脚本可见 |
| **IE2Video**（2025）[论文](https://arxiv.org/html/2512.05240) | 比较 RGB 化的 HyperE2VID 与 learned event encoder + LoRA 注入 LTX 的方案 | 初始图像 + 后续事件 → 对应 RGB 视频；可参考条件注入位置和表示消融 | **未确认官方 GitHub** |
| **LongE2V**（SIGGRAPH 2026）[论文](https://arxiv.org/abs/2607.08770) | 视频扩散以事件条件、锚帧和历史 context 进行长序列 chunk 展开 | 展开持续使用相应事件；需把 reconstruction / interpolation / prediction 各模式的条件读清 | [cdfan0627/LongE2V](https://github.com/cdfan0627/LongE2V)，模型、训练与推理说明可见 |
| **UniE2F**（2026）[论文](https://arxiv.org/abs/2602.19202) | Event condition 注入视频基础模型，结合 event-based inter-frame residual guidance；调制采样用于重建／插帧等模式 | 对应视频生成使用事件观测，不能仅按名称中的 prediction 视作 history-only 未来预测 | [CS-GangXu/UniE2F](https://github.com/CS-GangXu/UniE2F)，`src`、训练与推理脚本可见 |

对当前项目，这一组最有用的两个对照是：**事件 → 重建图像 → 冻结视觉 encoder**，以及 **事件 → 直接对齐视觉 latent**。前者可量化重建环节是否引入了不利于未来任务的误差；视觉质量更好不必然意味着预测特征更好。

### 6.3 事件模拟、配对数据与传感器模型

| 工具／论文 | 能提供什么 | 适用边界 | GitHub |
|---|---|---|---|
| **ESIM**（CoRL 2018）[论文](https://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) | 场景／轨迹渲染与自适应采样耦合，生成带时间戳的事件 | 环境动力学来自外部仿真或轨迹 | [uzh-rpg/rpg_esim](https://github.com/uzh-rpg/rpg_esim) |
| **Video to Events / vid2e**（CVPR 2020）[项目](https://github.com/uzh-rpg/rpg_vid2e) | 视频插帧后通过事件模型转换，复用视频数据 | 插值不能恢复原视频未记录的真实高速运动 | [uzh-rpg/rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) |
| **v2e**（CVPR Workshops 2021）[官方说明](https://github.com/SensorsINI/v2e) | 视频转 DVS，建模阈值差异、带宽、噪声等 | 可用于合成监督和域差异控制，仍需真实事件验证 | [SensorsINI/v2e](https://github.com/SensorsINI/v2e) |
| **DVS-Voltmeter**（ECCV 2022）[论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670571.pdf) | 随机电压过程与越阈值事件生成 | 参考噪声与触发机制；不是场景语义模型 | [Lynn0306/DVS-Voltmeter](https://github.com/Lynn0306/DVS-Voltmeter) |
| **Beyond Discretization**（2025）[论文](https://arxiv.org/abs/2504.02803) | 连续时间神经形态像素随机模型、精确模拟及解析统计 | 对连续事件时间建模有价值；非通用视觉 encoder | **未确认官方 GitHub** |
| **GS2E**（NeurIPS 2025 Datasets & Benchmarks）[论文](https://arxiv.org/abs/2505.15287) | 稀疏多视角 RGB → 静态 3DGS → 相机轨迹渲染 → 事件 | 可研究几何一致性与视点泛化；静态场景不提供物体交互动力学 | [PKU-YuanGroup/GS2E](https://github.com/PKU-YuanGroup/GS2E)，当前所见主要为文档／资源，README 仍写完整代码整理中 |

本节来源主要为本机生成模型专题，代码链接于本轮回到论文、作者项目页或仓库核查。**未确认**表示没有核实到作者实现，不把第三方同名项目、论文列表或网页模板仓库当作代码。


<a id="dsec"></a>
## 7. DSEC 的几何与时间对齐

同一场景并不意味着 Event / RGB tensor 天然逐像素可比较。时间对齐确定事件窗口与图像代表的时段；几何对齐确定同一物体落在哪些像素／patch；表征对齐确定要匹配的语义、空间关系或运动；因果范围确定这些信息在预测截止时刻是否已可获得。

### DSEC 的坐标和时间事实

| 对象 | 实际含义 | 对训练对齐的影响 |
|---|---|---|
| `events/left/events.h5` | 原始左事件像素，`cam0`，640×480；坐标仍有畸变 | 与 remapped RGB / dsec-det tracks 的使用约定一致；不能默认等同 rectified event 视图。 |
| `events/left/rectify_map.h5` | `rectify_map[y,x]` 给出原始事件到 rectified event 坐标的映射 | 映射后为 `camRect0`，此时需要与同一坐标系的监督配对。 |
| `images/left/rectified/` | 1440×1080 的 rectified 左图像视图，`camRect1` | Event 与 RGB 的 stereo pairs 分别 rectified；二者都叫 rectified 仍不表示逐像素重合。 |
| dsec-det 的 `images/left/distorted/` | 官方脚本将 rectified RGB remap 到原始左 Event 视图 `cam0` | 是带标定变换的重采样，不是直接 resize。 |
| DSEC flow GT | 左事件相机的 rectified 坐标 `camRect0`，100 ms 间隔的位移场 | 不能直接叠到 `distorted` RGB，不能将 100 ms 标签未经定义用于 10 ms 预测评价。 |
| event timestamp | HDF5 的事件时间需要加 `t_offset`，才与图像处在同一时钟 | `ms_to_idx` 是提取事件的加速索引，时间边界仍要精确裁剪。 |
| image timestamp / exposure | 有统一 image timestamp，也有左右相机曝光起止信息 | 历史输入需要已完成曝光；构造严格未来监督时应避免目标曝光跨过输入截止点。 |

以上由 [DSEC 官方格式](https://dsec.ifi.uzh.ch/data-format/)、[官方数据工具 GitHub](https://github.com/uzh-rpg/DSEC)、[dsec-det GitHub](https://github.com/uzh-rpg/dsec-det) 核实。实际 remap 的入口为 [remap_images_to_events.py](https://github.com/uzh-rpg/dsec-det/blob/master/scripts/remap_images_to_events.py)，映射计算见 [remapping.py](https://github.com/uzh-rpg/dsec-det/blob/master/src/dsec_det/remapping.py)。

**Remap 的几何边界。** dsec-det 的变换使用相机内参和旋转关系，没有逐像素深度和平移视差补偿；这是无限远近似，近距离对象和遮挡边界可能仍错位。官方 [Issue #25](https://github.com/uzh-rpg/DSEC/issues/25) 也明确讨论 infinite-depth 映射。因此，dense feature alignment 可以考虑有效区域 mask、几何可见性／置信度权重或局部匹配，但这些是本项目的候选处理，不能视为已有实验结论。

**对 Event-only forecasting 的直接约束。** 若截止为 $t$，每个事件窗口、recurrent state 和 token 都只能由 $E_{\le t}$ 构建。目标时刻的 RGB / Event 可以提供监督，却不能先与历史一起送入双向 encoder 再取历史 tokens；SCER 双侧累积、重建目标帧期间的事件，以及用后续事件更新的帧间检测均需按其原任务解释。空间 crop、resize、flip 应同步作用于事件、图像、标签和有效 mask，dense feature loss 则应明确 patch 对应和空事件区的权重。

本机相关记录：`/root/papers/notes/surveys/event-rgb-world-model-experiment-design-2026-10-10.md` §数据协议；`/root/papers/notes/surveys/event-world-action-research/research_protocol.md` DSEC 段；`/root/papers/projects/event-rgb-wm/experiments.md` E001 的 distorted RGB 与 tracks 路径。官方事实核实日期：2026-11-11；未运行这些论文训练或对齐实验。

<a id="project"></a>
## 8. 对当前项目的具体参考顺序

### 8.1 可直接接到当前 RGB 基线的组件

下表是基于方法接口的选型建议；本仓库目前实现的是 RGB DINO-Foresight 基线，尚未实现或验证表中的 Event 候选。

| 优先级 | 参考工作 | 值得先做的比较 |
|---|---|---|
| 1：明确输入表示 | EST、GEP、ScalE、RVT | 在相同历史时长／截止时刻下，比较计数伪帧、signed voxel、正负分离多 bin；记录分辨率、bin 数和归一化 |
| 1：跑通直接特征迁移 | GEP、ScalE、REALM、Network Grafting | Event encoder + adapter 对齐冻结 DINO patch 特征；逐点损失与空间关系损失分别消融 |
| 1：给出强 Event 对照 | RVT、F³ | 原生事件编码＋相同 future predictor／task head，与只做静态 RGB 对齐的方案比较 |
| 2：检查时间信息收益 | EvT、FLEET、SSM、HMNet | 在统一时间覆盖下比较细时间编码、recurrent state 和有限 memory；检查是否只是增加历史或算力 |
| 2：检验预测是否塑造编码器 | GEP、F³、E-Motion | 分开比较冻结 encoder 只训 predictor、冻结 predictor 让未来损失仅更新 encoder 与读出、encoder 与 predictor 联合训练 |
| 3：增加辅助运动或重建目标 | E-RAFT、EV-FlowNet、E2VID、EvFRA、EDFilter | 判断运动监督、重建桥梁或去噪是否改善未来任务，同时保留简单版本作为对照 |

对于“事件状态需要保留什么”，可以把完整状态与 RGB 对齐读出分开作为候选：

$$
h_t=E_\theta(\mathcal E_{\le t}),\qquad s_t=A_\psi(h_t),\qquad
\widehat h_{t+\Delta}=P_\phi(h_{\le t},\Delta).
$$

这里 $h_t$ 表示事件历史状态，$s_t$ 为与 RGB 教师比较的读出；当前对齐和未来监督可分别写为：

$$
\mathcal L_{\rm now}=d\!\left(s_t,\operatorname{sg}(T(I_t))\right),\qquad
\mathcal L_{\rm future}=d\!\left(A_\psi(\widehat h_{t+\Delta}),\operatorname{sg}(T(I_{t+\Delta}))\right).
$$

这是组织候选方案的示意式，不对应某篇论文的完整损失，也不表示已有实证优势。设置独立读出是为了检验：事件的细时间信息是否应该全部压进 RGB 教师空间，还是可以保留一部分供动态预测使用。静态对齐、未来对齐和原生事件预测需要分别比较。

### 8.2 RGB 基础模型与预测器参考

这些是教师／预测器接口参考，单列以免混入事件编码方法。

| 工作 | 参考接口与判断依据 | 论文／GitHub |
|---|---|---|
| **DINOv2 / DINOv3** | 图像预训练的 dense patch 特征；分别是 GEP 与 ScalE 的主要教师来源。教师版本、层、patch stride 和是否带 registers 会改变对齐目标 | [DINOv2论文](https://arxiv.org/abs/2304.07193) · [DINOv2 GitHub](https://github.com/facebookresearch/dinov2) · [DINOv3官方仓库及论文入口](https://github.com/facebookresearch/dinov3) |
| **DINO-Foresight** | 冻结图像 encoder，在特征空间学习未来预测，再由任务头读取；本仓库基线还包含 PCA / inverse PCA 接口 | [论文](https://arxiv.org/abs/2412.11673) · [Sta8is/DINO-Foresight](https://github.com/Sta8is/DINO-Foresight) |
| **DINO-world / Back to the Features** | 在 DINOv2 空间从过去帧预测未来特征，可再做动作条件适配；用于参考通用视频动态建模 | [论文](https://arxiv.org/abs/2507.19468) · [作者页面](https://baldassarrefe.github.io/publication/dino-world-2025/)；未确认官方 GitHub |
| **DINO-WM** | DINO patch 状态＋动作条件转移＋规划；可参考空间特征状态，但当前项目无动作输入 | [论文](https://arxiv.org/abs/2411.04983) · [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)。与上一行是不同工作 |
| **V-JEPA / V-JEPA 2** | 视频 encoder、target encoder 与 masked latent prediction；默认掩码预训练不等于只看过去的 forecasting。若替换图像教师，需在编码前隔离历史与未来，并避免跨 cutoff 的 tubelets | [V-JEPA论文](https://arxiv.org/abs/2404.08471) · [facebookresearch/jepa](https://github.com/facebookresearch/jepa)；[V-JEPA 2论文](https://arxiv.org/abs/2506.09985) · [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| **MC-JEPA** | 共享 encoder 联合学习 content 与 motion / optical flow；提供语义与运动共同塑造表示的参考 | [论文](https://arxiv.org/abs/2307.12698)；未确认官方 GitHub |

### 8.3 从文献比较中应保留的变量

1. **输入边界：**Event-only、RGB–Event fusion、训练 RGB／测试 Event、训练 Event／测试 RGB 分开列；未来目标模态与历史输入模态也分开。
2. **时间协议：**相同 cutoff、历史跨度、预测时长；区分固定事件数与固定时间；记忆 warm-up 和 state reset 纳入协议。
3. **监督资源：**外部 RGB/VFM 权重、配对数据、合成事件、任务标签分别记账；预训练数据更多与对齐算法更好不是同一结论。
4. **对齐对象：**全局语义、dense patch、局部几何、空间关系、运动或未来状态；损失降低只说明对应目标拟合改善。
5. **独立任务：**当前感知与 future detection／位置预测分别评估；状态延续、简单运动外推和强 Event 模型均有助于解释预测贡献。
6. **与 RGB 的比较：**明确比较的是 RGB 输入教师、直接喂事件的预训练模型，还是另一个 Event 模型。Event baseline 的提升不能换称“达到 RGB 教师水平”。

<a id="sources"></a>
## 9. 本机来源与覆盖核对

### 9.1 来源索引

| 本机材料（均位于 `/root/papers`） | 本文用途 |
|---|---|
| `index.md`、`md/Event/*/auto/*.md` | Event 分类已收录的25篇全文与元数据；核心方法和应用条目的主要材料 |
| `notes/surveys/event-rgb-vjepa2-design-2026-10-10/prior-work.md` 及 `sources/` | EST、RAMNet、EvT、AEGNN、EFNet、RVT、SSM、DAGr、PEPR 等编码与融合线索 |
| `notes/surveys/event-generative-world-model-survey-2026-10-10.md` | 事件生成、补全、重建、预测与模拟工具；区分输出对象和时间条件 |
| `notes/surveys/event-research/paper_matrix.md`、`report.md`、`datasets.md`、`sources/` | 驾驶、TTC、机器人操作／导航方法，数据和旧代码核查快照 |
| `notes/surveys/event-world-action-research/literature_matrix.md`、`sources.md`、`sources/` | F³、GEP、E-Motion、UniDynamics、ECHO、Event-VLA、DMR 等输入及预测机制 |
| `notes/surveys/event-world-action-research/motivation-and-few-shot-wm.md` | 事件知识迁移与少样本学习线索 |
| `notes/surveys/vjepa-dinov2-world-model-encoders-2026-11-03.md` | RGB/VFM 的预训练目标、teacher 与 predictor 分工 |
| `projects/event-rgb-wm/README.md`、`research.md`、`experiments.md` | 当前 Event-only／无动作范围，DSEC 协议和已有 RGB 基线状态 |

上述是本机来源定位，未随本文件复制到代码仓库；正文论文与 GitHub 链接可独立访问。本次回到官方原文和代码页补核，并将引用链补充与已有独立专题区分；没有将所有论文 references 展开成整个事件视觉领域的目录。

### 9.2 Event 分类25篇的覆盖核对

| 本机已收录论文 | 篇数 | 本文位置 |
|---|---:|---|
| ScalE、GEP | 2 | 第4节，视觉知识迁移 |
| F³、UniDynamics、ECHO | 3 | 第5.1节，预测目标与表示 |
| DMR | 1 | 第3节，任务相关表示分解 |
| FLEET、EventDrive、Event-VLA、E-VLA、DRFuser、eNavi | 6 | 第5.2节，token接口与任务融合 |
| DDD17、Event Steering、DDD20、Day–Night Domain Gap、Energy-Aware IL、RTRRL Driving、Human-Robot Navigation、SEBVS、Neuromorphic Eye-in-Hand、Real-Time Grasping、Event-Aided TTC、EvTTC、eAP | 13 | 第5.3节，其他应用与数据 |

其余条目来自既有专题、项目讨论或上述论文的引用链补查。例如 EST／RVT／RAMNet 来自编码专题，E-Motion／EvFRA／LongE2V 来自生成与世界模型专题，REALM 来自此前项目讨论并在本轮重读原文，EventSAM／ECDDP／OpenESS／Depth AnyEvent 等来自相关工作线索。这样可以追溯本库覆盖，同时区分方法精读与补充索引。
