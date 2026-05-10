# 第四阶段手工验收流程

适用基线：截至 2026-05-10 的当前仓库实现。以下流程用于人工确认阶段 4 的预算、恢复、审批与可观测性体验。

## 前置条件

- 已执行 `uv sync --dev`。
- 已设置 `DEEPSEEK_API_KEY`。
- 当前工作目录位于仓库根目录。
- 使用仓库内 `config.yaml`，其中已显式开启阶段 4 预算与详细审批 UI。

## 流程 1：预算内完成简单开发诊断

目标：验证 agent 能在预算内完成读上下文与命令诊断，并在 verbose 中展示阶段 4 事件。

步骤：

1. 运行：`uv run python -m linux_agent --config config.yaml --verbose "运行 pytest 并总结失败原因"`
2. 观察 stderr 中是否出现 `plan_update`、`reflection_scored`、`budget_warning`（若预算接近阈值）以及最终 `run_end`。
3. 观察 stdout 最终回答是否包含命令摘要和失败原因，而不是仅回显工具原始输出。

通过标准：

- run 成功结束，不出现无限循环。
- 若预算未耗尽，最终 `run_end.status` 为 `completed` 或其他非预算耗尽状态。
- verbose 输出能看见当前 reflection score、outcome 和预算摘要。

## 流程 2：审批复查与恢复

目标：验证审批卡片、pending run 复查、审批备注和回滚命令展示。

步骤：

1. 运行：`uv run python -m linux_agent --config config.yaml "请修改 README 中的一处示例文案"`
2. 确认进程以退出码 `2` 结束，并记录输出中的 `<run_id>`。
3. 复查审批详情：`uv run python -m linux_agent --config config.yaml --show-pending-run <run_id>`
4. 检查审批卡片中是否显示风险等级、影响文件、diff 预览、回滚命令、建议验证命令和预算剩余。
5. 使用备注批准：`uv run python -m linux_agent --config config.yaml --resume-run <run_id> --approve --decision-note "diff 范围已核对"`

通过标准：

- `--show-pending-run` 可独立渲染审批详情，而不触发恢复执行。
- 审批卡片不泄露完整原始写入内容，只显示裁剪后的 diff 预览。
- 批准后会继续写入并要求验证，而不是直接宣告成功。

## 流程 3：连续错误恢复后安全停止

目标：验证同一失败指纹不会无限重试，并在恢复预算耗尽后清晰停止。

步骤：

1. 临时把 `config.yaml` 中 `max_recovery_attempts_per_issue` 改为 `1`。
2. 运行一个会重复失败的目标，例如：`uv run python -m linux_agent --config config.yaml --verbose "检查 missing-dir 并告诉我里面有什么"`
3. 观察 verbose 中是否出现 `reflection_scored`、`recovery_attempted`，随后出现 `recovery_exhausted` 或 `budget_exhausted`。
4. 观察最终回答是否明确说明停止原因，而不是继续重复同样的命令。

通过标准：

- 对同一失败模式最多只进行有限次数恢复尝试。
- 最终回答包含恢复停止原因和当前阻塞点。
- `run_end` 中能看到 `budget_stop_reason=max_recovery_attempts` 或等价停止语义。

## 流程 4：预算耗尽安全停止

目标：验证命令或运行时预算耗尽时，系统不会继续扩张执行。

步骤：

1. 将 `config.yaml` 中 `max_command_count` 临时改为 `1`。
2. 运行一个至少需要两次命令/工具迭代的目标，例如：`uv run python -m linux_agent --config config.yaml --verbose "运行 pytest，然后再运行 ruff，并总结结果"`
3. 观察 verbose 是否先出现 `budget_warning`，之后在新命令执行前出现 `budget_exhausted`。
4. 恢复 `config.yaml` 原始配置。

通过标准：

- 预算耗尽后不会再执行新的命令。
- 最终回答会保留已完成步骤、当前阻塞点和预算停止原因。
- `run_end` 中可以看到 `budget_usage`、`budget_remaining` 和 `budget_stop_reason`。