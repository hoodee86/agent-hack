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

# 批准或拒绝某个暂停的 run
uv run python -m linux_agent --config config.yaml --resume-run <run_id> --approve
uv run python -m linux_agent --config config.yaml --resume-run <run_id> --reject

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
- 每次写入都会记录 backup manifest 和审计事件；如需恢复，可执行 `--rollback-run <run_id>`。
- `--verbose` 下会额外看到 `approval_requested`、`write_applied`、`write_rollback` 的详细输出，包括目标文件、diff 预览、备份和回滚信息。

## 开发

```bash
uv run pytest          # 运行测试
uv run mypy src/       # 类型检查
```
