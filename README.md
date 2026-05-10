# linux-agent

基于 LangGraph 的受控 Linux workspace Agent 原型。

当前默认使用 `deepseek-v4-pro`，通过 OpenAI 兼容的 `ChatOpenAI` 接口连接 `https://api.deepseek.com`。

当前已支持：

- 只读文件系统探索：`list_dir`、`read_file`、`search_text`
- 受控开发命令执行：`run_command`
- 受控文本写入：`apply_patch`、`write_file`（默认需要显式审批）
- 命令执行审计、verbose 控制台明细、以及命令结果摘要
- 命令失败后的继续诊断：可先跑 `pytest` / `mypy`，再读取相关文件继续总结
- 审批暂停 / 恢复、写操作审计、备份 manifest 与按 run id 回滚
- 写后强制验证，以及验证失败时的自动回滚摘要
- 结构化计划更新、reflection score、有界恢复状态与预算硬熔断
- 更完整的审批卡片、`--show-pending-run` 复查，以及带备注的审批响应

## 快速开始

```bash
# 设置 DeepSeek API Key
export DEEPSEEK_API_KEY="<your-api-key>"

# 安装依赖
uv sync --dev

# 使用仓库内 config.yaml 运行
uv run python -m linux_agent --config config.yaml "项目里有哪些 Python 文件"

# 或直接传 workspace
uv run python -m linux_agent --workspace /path/to/project "README 里说了什么"

# 运行测试并让 Agent 总结失败原因
uv run python -m linux_agent --config config.yaml "运行测试并总结失败原因"

# 在 stderr 中查看分段彩色调试日志
uv run python -m linux_agent --config config.yaml --verbose "README 里说了什么"

# 运行失败测试，并在 verbose 模式下查看命令 proposal / policy / result / summary
uv run python -m linux_agent --config config.yaml --verbose "运行 pytest 并总结失败原因"

# 运行类型检查，并要求 Agent 给出下一步建议
uv run python -m linux_agent --config config.yaml --verbose "运行 mypy 并给出下一步建议"

# 额外打印每轮发给模型的 system/history/user 消息序列
uv run python -m linux_agent --config config.yaml --verbose --show-prompts "README 里说了什么"

# 当写操作需要审批时，CLI 会以退出码 2 结束，并输出 resume 指令
uv run python -m linux_agent --config config.yaml "请修改 README 中的示例命令"

# 重新查看某个 pending run 的审批详情
uv run python -m linux_agent --config config.yaml --show-pending-run <run_id>

# 批准或拒绝某个暂停的 run，并可记录审批备注
uv run python -m linux_agent --config config.yaml --resume-run <run_id> --approve
uv run python -m linux_agent --config config.yaml --resume-run <run_id> --approve --decision-note "验证范围足够窄"
uv run python -m linux_agent --config config.yaml --resume-run <run_id> --reject --decision-note "diff 过大，需要缩小改动面"

# 按 run id 回滚之前的写操作
uv run python -m linux_agent --config config.yaml --rollback-run <run_id>
```

## 阶段 2 命令示例

典型目标：

- `运行测试并总结失败原因`
- `运行 mypy 并给出下一步建议`
- `运行 ruff check 并说明最值得先处理的问题`

`--verbose` 下，控制台会显示命令 proposal、策略决策、执行结果和 run end summary，例如：

```text
[linux-agent] Iteration 1 | Tool Proposal
Command: uv run pytest -q
Working Directory: .

[linux-agent] Iteration 1 | Policy Guard
Decision: allow

[linux-agent] Iteration 1 | Tool Result
Exit Code: 1
Truncated: False
Stdout Preview: ... failed, 3 passed

[linux-agent] Run End
Command Summaries:
- uv run pytest -q [cwd=.] -> failed (exit 1)
```

## 阶段 2 边界

- 只支持 allowlist 内的开发命令前缀，例如 `pytest`、`mypy`、`ruff`、`git status`、`git diff`。
- 不支持任意 shell 语法：如管道、重定向、命令替换、`&&` / `||` / 后台执行。
- 不支持交互式命令、后台常驻进程、联网下载或安装依赖。
- `cwd` 只能落在 `command_working_dirs` 声明的 workspace 子目录内。

## 阶段 3 审批写入

- 写操作不会直接执行；当模型提出 `apply_patch` / `write_file` 时，当前进程会输出审批摘要并以退出码 `2` 结束。
- 审批状态会持久化到 `log_dir/state/<run_id>.json`，可通过 `--resume-run <run_id> --approve|--reject` 恢复或拒绝。
- 审批通过后的写入不会直接宣告成功；agent 必须先运行一次验证命令，如测试、lint 或类型检查。
- 若模型在写后直接尝试结束，graph 会阻断这次完成并返回“仍需验证”的结果。
- `auto_rollback_on_verify_failure=true` 时，验证失败会自动基于 manifest 回滚最近一次写入；如需手工恢复，也可执行 `--rollback-run <run_id>`。
- `--verbose` 下会额外看到 `approval_requested`、`write_applied`、`write_rollback`、验证状态、验证命令和 rollback 结果。

一次完整的阶段 3 流程通常是：读取上下文 -> 提出 `apply_patch` / `write_file` -> CLI 暂停等待审批 -> `--resume-run <run_id> --approve` -> 自动执行写入 -> 自动要求验证命令 -> 根据验证结果总结完成或回滚。

## 阶段 4：预算、恢复与审批体验

- `config.yaml` 现在可以显式配置 `max_command_count`、`max_runtime_seconds`、`max_plan_revisions`、`max_recovery_attempts_per_issue`、`budget_warning_ratio`、`reflection_replan_threshold`、`reflection_stop_threshold`、`approval_ui_mode`。
- Planner 会看到剩余预算；当命令数、运行时长、计划修订次数或恢复次数逼近阈值时，运行中会先发 `budget_warning`，彻底耗尽后会安全停止并在最终回答中总结当前进度与阻塞点。
- Reflector 会把每次 observation 归纳成 `last_reflection` 和 `recovery_state`，支持 `continue`、`retry`、`replan`、`stop` 四类有界后续动作。
- `--verbose` 下，控制台会额外打印 `plan_revised`、`reflection_scored`、`recovery_attempted`、`recovery_exhausted`、`budget_warning`、`budget_exhausted`、`approval_presented`、`approval_response` 等阶段 4 事件。
- 审批暂停后，除了直接 approve / reject，还可以通过 `--show-pending-run <run_id>` 复查风险等级、影响文件、diff 预览、回滚命令、建议验证命令、剩余预算和当前恢复状态。

常见停止语义：

- 命令预算耗尽：不会再执行新的命令，会在最终回答里说明已完成步骤、阻塞点和预算原因。
- 运行时长耗尽：会直接结束当前 run，并保留预算快照与最后一次 reflection。
- 恢复次数耗尽：同一失败指纹不会无限重试，最终会以 `max_recovery_attempts` 停止。

## 手工验收

- 手工验收流程见 [docs/phase4_manual_acceptance.md](docs/phase4_manual_acceptance.md)。
- 建议至少演练 3 条链路：预算内完成任务、审批复查与恢复、连续错误恢复后安全停止。

## 开发

```bash
uv run pytest          # 运行测试
uv run mypy src/       # 类型检查
```
