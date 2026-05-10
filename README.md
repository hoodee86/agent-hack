# linux-agent

基于 LangGraph 的只读 Linux Agent 原型（第一阶段）。

## 快速开始

```bash
# 安装依赖
uv sync

# 运行（需要设置 LINUX_AGENT_WORKSPACE 或传 --workspace）
uv run python -m linux_agent --workspace /path/to/project "项目里有哪些 Python 文件"
```

## 开发

```bash
uv run pytest          # 运行测试
uv run mypy src/       # 类型检查
```
