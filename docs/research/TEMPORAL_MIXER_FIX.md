# TemporalMixer 大批量 CUDA 修复与分支迁移

## 来源与影响范围

`TemporalMixer` 首次由 `codex/event-wm-fusion` 的 `d66a8f51bb87c377f2a18ff3e0ba003acfee0435` 引入。后续 transition、motion-rollout、transition-diagnostic 继承的公共实现相同。

事件编码把输入变成 `[B × intervals × spatial_sites, bins, hidden_dim]` 后执行时间注意力。四步诊断的 `B=8`、`intervals=4`、空间网格 `32×64`，因此有效 attention batch 为 **65,536**。该形状触及部分 CUDA SDPA 后端的 batch/grid 限制，与服务器报告及 [PyTorch issue #146704](https://github.com/pytorch/pytorch/issues/146704) 一致；具体运行选择的后端未做 profiling。

修复放在最早引入问题的 fusion 分支；transition、motion-rollout、transition-diagnostic 逐级 rebase。旧 Event-only 分支没有这个公共组件，不在此次改动范围内。问题同样可能由较大的历史事件 batch 或 `HistoryFeatures` 触发，并非只有四步诊断受影响。

## 修复方式与兼容性

公共 `TemporalMixer` 沿独立的 `batch_sites` 维自动分块，每次最多 **16,384** 个 site；每块完整保留时间 bins，再按原顺序拼接。LayerNorm、attention、FFN 都不跨 site 交互，因此保持原有时间建模关系和反向传播路径。

参数、state_dict、模型规格和训练配置不变，已有 checkpoint 可严格恢复。不需要降低数据 batch、缩短预测跨度、重新训练或改变全局 attention 后端。不同 kernel/batch 组织可能存在正常浮点舍入差异。

## 回归验证

```bash
python3 -m pytest -q -rs tests/test_research_temporal_chunking.py
python3 -m pytest -q -rs
```

新增测试检查旧公式与分块计算的前向及梯度一致性、65,536 附近的实际 batch 边界、Event 编码和历史融合调用方，以及旧 state_dict 加载。CUDA 可用时运行实际 `[65536, 5, 128]` 输入的 FP32/BF16 回归；无 CUDA 时这些 GPU 测试明确跳过。

服务器还应使用原 M-explicit checkpoint，重新运行四步诊断的 **batch 8、32 个样本 smoke**。沿用原命令，将输出目录换成新目录，并加入 `--max-samples 32`；检查完整运行及各路径指标，再按需要跑全量。不覆盖已有 batch 2 结果。

## 服务器独有实验分支

GitHub 上更新的是四个 `codex/event-wm-*` 开发分支。反馈中的 `codex/wm-run` 和 `codex/wm-diag-run` 只在实验服务器 bare 仓库存在；以下命令保留各自的实验配置、修复与结果提交。它们用于 clean worktree，执行前先停止使用该 worktree 的训练/评价进程。

```bash
git fetch https://github.com/hanzh0816/event_foresight.git \
  refs/heads/codex/event-wm-motion-rollout:refs/remotes/wm-fix/event-wm-motion-rollout \
  refs/heads/codex/event-wm-transition-diagnostic:refs/remotes/wm-fix/event-wm-transition-diagnostic
```

在 `codex/wm-run` 所在 worktree 执行：

```bash
git switch codex/wm-run
git branch backup/wm-run-before-temporal-fix
git merge-base --is-ancestor 4a7dc5e37d0150a590d1b90f69e9a0ae1e2e76ea HEAD && \
git rebase --onto refs/remotes/wm-fix/event-wm-motion-rollout \
  4a7dc5e37d0150a590d1b90f69e9a0ae1e2e76ea
```

在 `codex/wm-diag-run` 所在 worktree 执行：

```bash
git switch codex/wm-diag-run
git branch backup/wm-diag-run-before-temporal-fix
git merge-base --is-ancestor cebf33fb29fbefbc54c5202b32edc77e7c32714e HEAD && \
git rebase --onto refs/remotes/wm-fix/event-wm-transition-diagnostic \
  cebf33fb29fbefbc54c5202b32edc77e7c32714e
```

完成测试后，使用指向服务器 bare 仓库的真实 remote 推送，`--force-with-lease` 显式指定该远程更新前的 branch SHA。两条实验分支各自重放自己的提交，不互相覆盖。已有报告与 checkpoint 中记录的旧 commit 是其真实运行来源，保持原值；本次修复后的复跑另记新 commit。
