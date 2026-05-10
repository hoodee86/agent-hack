# linux-agent

基于 LangGraph 的只读 Linux Agent 原型（第一阶段）。

当前默认使用 `deepseek-v4-pro`，通过 OpenAI 兼容的 `ChatOpenAI` 接口连接 `https://api.deepseek.com`。

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
```

## 开发

```bash
uv run pytest          # 运行测试
uv run mypy src/       # 类型检查
```
