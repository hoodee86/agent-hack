# linux-agent

基于 LangGraph 的受控 Linux workspace Agent 原型。

当前默认使用 `deepseek-v4-pro`，通过 OpenAI 兼容的 `ChatOpenAI` 接口连接 `https://api.deepseek.com`。

当前已支持：

- 只读文件系统探索：`list_dir`、`read_file`、`search_text`
- 受控开发命令执行：`run_command`
- 命令执行审计、verbose 控制台明细、以及命令结果摘要

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

# 额外打印每轮发给模型的 system/history/user 消息序列
uv run python -m linux_agent --config config.yaml --verbose --show-prompts "README 里说了什么"
```

## 开发

```bash
uv run pytest          # 运行测试
uv run mypy src/       # 类型检查
```
