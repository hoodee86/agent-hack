# 第二、三阶段任务详细拆解：受控命令执行与受控写操作

> 对应设计文档：`langgraph_linux_agent_design.md` § 10 阶段 2 / 阶段 3
>
> 适用基线：截至 2026-05-10 的当前仓库实现。阶段 1 已完成，默认 LLM 为 `deepseek-v4-pro`，通过 OpenAI 兼容接口接入 `https://api.deepseek.com`，并使用原生 tool calling 驱动 `list_dir` / `read_file` / `search_text`。

## 文档目标

本文件把设计文档中的“阶段 2：受控命令执行”和“阶段 3：受控写操作”拆解成可实施、可测试、可验收的任务清单，作为后续开发的详细设计基线。

**当前实现状态（2026-05-10）**：

- T18-T25 已完成。
- T26-T35 未开始。

本文默认遵循以下原则：

- 继续沿用当前 `graph.py` 中的原生 tool calling，不回退到结构化输出中间层。
- 优先扩展现有 `Planner -> PolicyGuard -> ToolExecutor -> Reflector -> Finalizer` 主路径，只在阶段 3 为审批流增加必要分支。
- 在阶段 2/3 初版中，尽量保持 `AgentState` 和 `Observation` 的兼容性，避免为支持命令执行或写操作而推翻阶段 1 已验证的消息流和审计流。
- 所有新能力都必须接入现有审计日志与 verbose 输出，而不是另起一套调试通道。

## 阶段目标

### 阶段 2：受控命令执行

目标是让 Agent 在受控 workspace 内执行测试、lint、类型检查、构建等开发命令，并把失败信息转化为下一轮推理输入，而不是仅停留在文件阅读能力上。

**阶段验收标准**：Agent 可以运行测试或 lint 命令，读取失败输出，在必要时继续读取相关文件，并给出下一步建议。

### 阶段 3：受控写操作

目标是让 Agent 在人工确认后，对 workspace 中文本文件进行可审计、可回滚的修改，并在修改后自动运行验证命令。

**阶段验收标准**：Agent 可以先读取上下文、提出 patch、等待用户确认、应用变更、运行测试验证，并在失败时保留回滚路径和差异摘要。

## 关键设计决策

### 1. 命令执行与写操作分阶段推进

- 阶段 2 只引入 `run_command`，不引入任意写文件工具。
- 阶段 3 先实现 `apply_patch` 作为主要写接口，再评估是否补充受限版 `write_file`。
- 所有“安装依赖”“联网下载”“修改系统服务”类操作不属于阶段 2/3 的默认自动允许范围。

### 2. 保持现有 Observation 结构稳定

当前仓库中的 `Observation` 采用：

```python
class Observation(TypedDict):
    tool: str
    ok: bool
    result: dict[str, object] | None
    error: str | None
    duration_ms: int
```

阶段 2/3 建议沿用这一结构，把命令执行结果、diff 摘要、备份信息等放进 `result` 子对象，避免一次性重构全部图节点和阶段 1 测试。

### 3. 阶段 3 默认采用审批暂停/恢复模型

- `apply_patch` / `write_file` 默认不直接执行。
- Policy Guard 对写操作返回 `needs_approval`。
- 图进入“暂停并输出审批请求”的路径。
- 用户批准后，再从持久化状态恢复继续执行。

这比在 CLI 内直接做同步交互更容易审计，也更适合未来接 HTTP API 或 UI。

## 任务总览

```text
阶段 2：受控命令执行
T18 配置扩展（命令策略与资源限制）
T19 命令解析与风险分类（policy.py）
T20 run_command skill（skills/shell.py）
T21 graph 集成与工具 schema（graph.py）
T22 审计与 verbose 输出增强（audit.py / app.py）
T23 Reflector / Finalizer 命令结果利用
T24 阶段 2 单元测试与安全回归
T25 阶段 2 集成验收与文档更新

阶段 3：受控写操作
T26 配置与状态扩展（审批模型）
T27 写操作策略与审批决策（policy.py）
T28 apply_patch skill（推荐 MVP）
T29 write_file skill（受限补充）
T30 审批暂停 / 恢复链路（graph.py / app.py）
T31 diff、变更摘要与审计落盘
T32 备份、原子写入与回滚
T33 Planner / Reflector 写操作闭环
T34 阶段 3 单元测试、集成测试、安全回归
T35 阶段 3 验收与文档收口
```

依赖关系：

- 阶段 2：`T18 -> T19 -> T20 -> T21/T22 -> T23 -> T24 -> T25`
- 阶段 3：`T26 -> T27 -> T28/T29 -> T30 -> T31/T32 -> T33 -> T34 -> T35`

---

## 阶段 2 任务拆解

### T18 — 配置扩展（命令策略与资源限制）

**文件**：`src/linux_agent/config.py`、`config.yaml`

**任务内容**：

- 在 `AgentConfig` 中新增命令执行相关配置：
  - `default_timeout_seconds`
  - `max_output_bytes`
  - `max_stderr_bytes`（可选；若不单列，则沿用统一上限）
  - `command_allowlist`
  - `command_denylist`
  - `command_approvallist`（阶段 2 可先不用，但字段预留给阶段 3/4）
  - `command_env_allowlist`
  - `command_working_dirs` 或等价的 cwd 限制配置
- YAML 配置支持相对路径解析，并给出保守默认值。
- 默认策略需要覆盖当前仓库的真实开发路径，例如：
  - `uv run pytest -q`
  - `uv run mypy src/`
  - `uv run ruff check .`
- 默认不允许 `sudo`、`su`、`sh -c`、`bash -c`、`python -c`、重定向、管道、后台执行、任意网络访问。

**验收标准**：

- `load_config()` 在不破坏现有阶段 1 行为的前提下，能返回完整命令执行配置。
- 示例 `config.yaml` 可以声明 allowlist / denylist / timeout 等字段。
- 未显式声明的命令执行参数都有保守默认值。

### T19 — 命令解析与风险分类（policy.py）

**文件**：`src/linux_agent/policy.py`（必要时可拆出 `command_policy.py`）

**任务内容**：

- 为 `run_command` 增加独立的命令解析逻辑，建议实现：
  - `parse_command(raw: str) -> list[str]`
  - `classify_command(argv: list[str], config: AgentConfig) -> Literal["low", "medium", "high"]`
  - `evaluate_command_call(...) -> Literal["allow", "deny"]`
- 使用 `shlex.split()` 或等价方案解析命令，并在策略层拒绝以下模式：
  - 空命令
  - `;`、`&&`、`||`、`|`、`>`、`>>`、`<`
  - 反引号、`$()`
  - 后台执行符号 `&`
- 对 `cwd` 进行 workspace 边界校验，禁止命令在 workspace 外执行。
- 对 `env` 做 allowlist 过滤，只允许显式白名单变量透传。
- 风险分级至少区分：
  - `low`：测试、lint、类型检查、只读开发命令
  - `medium`：可能修改 workspace 产物但仍在仓库边界内的构建命令
  - `high`：提权、删除、系统级命令、网络访问、shell 注入模式

**验收标准**：

- `uv run pytest -q`、`python -m pytest -q`、`uv run mypy src/` 等命令可以被解析和放行。
- `sudo ls`、`sh -c "pytest"`、`pytest -q > out.txt`、`rm -rf /` 等命令必须被拒绝。
- 任意 `cwd` 越界或命中敏感路径时必须被拒绝。

### T20 — run_command skill（skills/shell.py）

**文件**：`src/linux_agent/skills/shell.py`（新增）

**任务内容**：

- 新增 `run_command(command: str, config: AgentConfig, **kwargs) -> dict[str, object]`。
- 使用 `subprocess.run(..., shell=False)` 或等价安全接口执行命令。
- 支持输入参数：
  - `command`
  - `cwd`
  - `timeout_seconds`
  - `env`
- 返回结构建议至少包含：

```python
{
    "ok": bool,
    "command": str,
    "argv": list[str],
    "cwd": str,
    "exit_code": int | None,
    "stdout": str,
    "stderr": str,
    "duration_ms": int,
    "timed_out": bool,
    "truncated": bool,
}
```

- 对 stdout / stderr 做安全截断，避免上下文爆炸。
- 保留 UTF-8 解码失败时的兜底策略（如 `errors="replace"`）。
- 明确区分“命令执行失败”“命令超时”“策略已拒绝”三类错误。

**验收标准**：

- 成功命令能返回 exit code、stdout、stderr、duration。
- 失败命令能保留 stderr 与退出码。
- 超时命令能被终止，并返回 `timed_out=true`。
- 超长输出不会撑爆日志或消息上下文。

### T21 — graph 集成与工具 schema（graph.py）

**文件**：`src/linux_agent/graph.py`、`src/linux_agent/state.py`

**任务内容**：

- 在 `_MODEL_TOOLS` 中新增 `@tool("run_command")` schema。
- Planner prompt 需要新增明确规则：
  - `run_command` 仅用于测试、lint、构建、类型检查、项目内安全诊断。
  - 每轮最多调用一个命令。
  - 优先使用结构化工具参数，不允许把 shell 拼接技巧交给模型自己发挥。
- Tool Executor 为 `run_command` 增加分发表项。
- `ToolCall.risk_level` 的计算要兼容命令场景。
- 继续保持当前消息序：`system -> history -> current human`。

**验收标准**：

- Planner 能提出 `run_command` 工具调用。
- Tool Executor 能执行 `run_command` 并把结果写回 `ToolMessage`。
- 不破坏现有 `list_dir` / `read_file` / `search_text` 的行为和测试。

### T22 — 审计与 verbose 输出增强（audit.py / app.py）

**文件**：`src/linux_agent/audit.py`、`src/linux_agent/app.py`

**任务内容**：

- 复用当前审计事件体系，增强 `tool_proposed` / `policy_decision` / `tool_result` 的命令字段：
  - `command`
  - `argv`
  - `cwd`
  - `exit_code`
  - `timed_out`
  - `truncated`
- verbose 输出要能直接看出：
  - 执行了什么命令
  - 在哪个目录执行
  - 退出码是多少
  - stdout / stderr 关键片段是什么
- 如果引入新事件类型（例如 `command_classified`），必须保证 JSONL 和控制台输出都能消费它。

**验收标准**：

- `--verbose` 模式能完整展示命令 proposal、策略决策和结果。
- JSONL 审计日志中可追踪每条命令的输入、决策和输出摘要。
- 超长输出在控制台只显示摘要，但在日志中仍可看到截断标记。

### T23 — Reflector / Finalizer 命令结果利用

**文件**：`src/linux_agent/graph.py`

**任务内容**：

- Reflector 需要理解命令退出码、stderr 和 stdout 摘要：
  - 测试失败时，优先引导 Planner 去读失败文件或定位报错位置。
  - 命令超时时，停止盲目重试并向用户报告。
  - 输出被截断时，提示模型先缩小范围再继续。
- Finalizer 需要总结：
  - 执行过哪些命令
  - 哪些命令失败
  - 失败证据来自 stdout/stderr 的哪些关键信息

**验收标准**：

- 运行失败测试后，最终回答能包含命令、退出码和错误摘要。
- 连续命令失败会正确触发熔断，而不是无限重试。
- 命令超时和输出截断都能在最终回答中被明确提及。

### T24 — 阶段 2 单元测试与安全回归

**文件**：`tests/` 下新增或扩展相关测试

**任务内容**：

- 新增 shell skill 单元测试：
  - 成功命令
  - 失败命令
  - 超时命令
  - 输出截断
- 新增策略测试：
  - allowlist / denylist
  - shell metacharacters 拒绝
  - cwd 越界拒绝
  - env 注入过滤
- 新增 graph 行为测试：
  - Planner 提出 `run_command`
  - ToolMessage 与 tool_call_id 顺序正确
  - Reflector 能从 stderr 摘要推进下一步

**验收标准**：

- 所有新增单测稳定可重复。
- 已有阶段 1 测试不回归。
- 安全回归用例覆盖危险命令、路径越界、shell 注入。

### T25 — 阶段 2 集成验收与文档更新

**文件**：`README.md`、`config.yaml`、相关 docs

**任务内容**：

- 增加至少两个端到端场景：
  - 运行失败测试并总结错误
  - 运行 lint / type check 并给出下一步建议
- 更新 README 中的 verbose 样例，展示命令执行明细。
- 在设计文档中标注阶段 2 的实现范围和已知限制，例如：
  - 不支持任意 shell 语法
  - 不支持交互式命令
  - 不支持后台常驻进程

**验收标准**：

- 文档能指导用户在真实仓库里安全运行阶段 2 Agent。
- 手工验收场景与自动化测试结论一致。

---

## 阶段 3 任务拆解

### T26 — 配置与状态扩展（审批模型）

**文件**：`src/linux_agent/config.py`、`src/linux_agent/state.py`

**任务内容**：

- 扩展 `risk_decision`：从 `Literal["allow", "deny"]` 升级为 `Literal["allow", "deny", "needs_approval"]`。
- 新增 `ApprovalRequest` 结构，建议包含：

```python
class ApprovalRequest(TypedDict):
    id: str
    tool: str
    args: dict[str, object]
    reason: str
    impact_summary: str
    diff_preview: str | None
    backup_plan: str | None
```

- `AgentState` 新增：
  - `pending_approval: ApprovalRequest | None`
  - 如需恢复执行，可增加 `resume_token` 或等价标识
- `AgentConfig` 新增写操作相关配置：
  - `write_requires_approval`
  - `max_patch_bytes`
  - `max_patch_hunks`
  - `backup_dir`
  - `auto_rollback_on_verify_failure`

**验收标准**：

- 阶段 1/2 在 `pending_approval is None` 时行为不变。
- 状态结构可以表达“等待审批”的中间状态。
- 所有新字段都有默认值和说明。

### T27 — 写操作策略与审批决策（policy.py）

**文件**：`src/linux_agent/policy.py`

**任务内容**：

- 把 `evaluate_tool_call()` 扩展为支持三类结果：
  - `allow`
  - `deny`
  - `needs_approval`
- 默认规则：
  - `apply_patch`、`write_file` 一律 `needs_approval`
  - workspace 外路径、敏感路径、二进制文件、超大 patch 直接 `deny`
- 审批理由要结构化输出，至少包括：
  - 为什么需要审批
  - 预期修改哪些文件
  - 是否涉及新增/删除/覆盖
- 命令执行与写操作策略要统一在一个入口评估，避免 graph 层写两套判断。

**验收标准**：

- 所有写工具默认不会被自动放行。
- 对越界路径或敏感路径的写请求必须直接拒绝。
- `policy_decision` 审计事件能记录审批原因。

### T28 — apply_patch skill（推荐 MVP）

**文件**：`src/linux_agent/skills/write.py`（推荐新增）

**任务内容**：

- 实现 `apply_patch` 作为阶段 3 的主写接口。
- 推荐支持统一 diff 或仓库内部约定 patch 格式，但必须满足：
  - 只能修改文本文件
  - 只能操作 workspace 内路径
  - 应用前先做 dry-run 校验
  - hunk 无法匹配时失败并返回明确错误
- 返回结构建议包含：

```python
{
    "ok": bool,
    "changed_files": list[str],
    "added_lines": int,
    "removed_lines": int,
    "diff": str,
    "backup_paths": list[str],
}
```

- 应优先支持 `Update File` / `Add File` 两类场景，`Delete File` 可放在后续版本评估。

**验收标准**：

- 能对 workspace 内文本文件应用小规模 patch。
- 对不存在上下文、二进制文件、越界路径、超大 patch 必须失败关闭。
- 返回结果中包含文件级变更摘要。

### T29 — write_file skill（受限补充）

**文件**：`src/linux_agent/skills/write.py` 或 `src/linux_agent/skills/filesystem.py`

**任务内容**：

- `write_file` 作为 `apply_patch` 的补充，而不是替代。
- 推荐 MVP 只支持以下模式：
  - `create_only`
  - `append`（可选）
  - `replace_range`（在 patch 能力成熟后再考虑）
- 所有模式都必须经过审批。
- 新文件创建要限制路径、后缀和大小，避免模型直接生成大块未知内容。

**验收标准**：

- 能在审批通过后创建新文件或追加有限内容。
- 不能覆盖 workspace 外或敏感文件。
- 与 `apply_patch` 相比，`write_file` 的适用边界在文档中清晰可见。

### T30 — 审批暂停 / 恢复链路（graph.py / app.py）

**文件**：`src/linux_agent/graph.py`、`src/linux_agent/app.py`，必要时新增持久化辅助模块

**任务内容**：

- 为图增加审批暂停路径。推荐方案：
  - `planner -> policy_guard -> approval_pause -> finalizer`
  - 批准后从持久化状态恢复，再进入 `tool_executor`
- CLI 初版建议支持：
  - 运行时遇到审批请求，输出审批摘要并退出特定状态码
  - 通过单独命令或参数恢复，例如 `--resume-run <run_id> --approve`
- 需要持久化待恢复状态，不能只依赖内存。
- 审批被拒绝时，要形成明确的最终回答并结束运行。

**验收标准**：

- 写操作请求会暂停而不是直接执行。
- 用户批准后，Agent 可以继续执行同一 run。
- 用户拒绝后，不会对文件系统产生任何修改。

### T31 — diff、变更摘要与审计落盘

**文件**：`src/linux_agent/audit.py`、`src/linux_agent/app.py`、写操作 skill

**任务内容**：

- 为写操作补充审计事件，建议至少包括：
  - `approval_requested`
  - `write_applied`
  - `write_rollback`
- 日志中记录：
  - 目标文件
  - 变更摘要
  - 行数增减
  - diff 预览
  - 审批请求 ID
- verbose 输出需要能读出：
  - 改了哪些文件
  - patch 预览是什么
  - 是否已经备份
  - 是否已经回滚

**验收标准**：

- 每次写操作都有完整的审计链路。
- 大 diff 在控制台中只显示预览，在日志中保留完整或更大额度的记录。
- 审批请求与实际写入事件能通过 request id 关联。

### T32 — 备份、原子写入与回滚

**文件**：`src/linux_agent/skills/write.py`、`src/linux_agent/config.py`

**任务内容**：

- 在写入前为受影响文件创建备份，建议放入：
  - `.linux-agent/backups/<run_id>/...`
  - 或 `config.backup_dir / <run_id> / ...`
- 使用临时文件 + 原子替换实现落盘，避免部分写入。
- 当以下情况出现时，保留或触发回滚：
  - patch 应用失败
  - 写入后验证命令失败且策略要求自动回滚
  - 恢复流程中用户要求撤销变更
- 维护备份 manifest，记录原路径与备份路径映射。

**验收标准**：

- 任意失败场景下都能定位备份并恢复原始内容。
- 原子写入不会留下半写入文件。
- 回滚事件能被日志和最终回答引用。

### T33 — Planner / Reflector 写操作闭环

**文件**：`src/linux_agent/graph.py`

**任务内容**：

- 在 Planner prompt 中新增写操作规则：
  - 只有在已有充分观察证据时才允许提出 patch。
  - patch 前应优先读取目标文件相关上下文。
  - 写后必须运行验证命令，不能直接宣告完成。
- Reflector 需要支持写后分支：
  - 验证通过：进入总结
  - 验证失败：读取错误、决定是否再修一轮或建议回滚
  - 用户拒绝审批：给出替代方案
- Finalizer 需要总结：
  - 变更了哪些文件
  - 是否经过审批
  - 验证命令结果如何
  - 若失败，当前可回滚路径是什么

**验收标准**：

- Agent 能形成“读上下文 -> 提 patch -> 等审批 -> 写入 -> 验证 -> 总结”的闭环。
- 写后不会跳过验证环节。
- 审批拒绝、验证失败、回滚三条路径都能形成可理解的最终回答。

### T34 — 阶段 3 单元测试、集成测试、安全回归

**文件**：`tests/` 下新增或扩展相关测试

**任务内容**：

- 单元测试覆盖：
  - `needs_approval` 判定
  - patch 解析与 dry-run
  - 原子写入
  - 备份和回滚
  - diff 统计
- 集成测试覆盖：
  - 读文件 -> 生成 patch -> 暂停审批
  - 批准后恢复执行并跑测试
  - 拒绝审批且不改文件
  - 验证失败后生成回滚建议或自动回滚
- 安全回归覆盖：
  - symlink 越界写入
  - 二进制文件 patch
  - 大 patch 攻击
  - 敏感文件路径写入

**验收标准**：

- 阶段 3 的所有核心风险点都有自动化测试。
- 阶段 1/2 测试保持绿色。
- 审批与回滚路径有端到端覆盖。

### T35 — 阶段 3 验收与文档收口

**文件**：`README.md`、`config.yaml`、相关 docs

**任务内容**：

- 更新 README，加入审批与恢复执行的使用说明。
- 更新设计文档，标明阶段 3 已实现的写操作范围：
  - 是否只支持 `apply_patch`
  - `write_file` 支持哪些模式
  - 自动回滚是否默认开启
- 增加运维视角说明：
  - 备份放在哪里
  - 如何手工恢复
  - 如何清理旧备份

**验收标准**：

- 用户能根据文档完成一次完整的“审批后修改并验证”流程。
- 文档中明确说明已支持与未支持的写操作边界。

---

## 跨阶段验收场景建议

建议至少保留以下 4 个标准场景作为开发里程碑：

1. **失败测试诊断**：Agent 运行 `uv run pytest -q`，读取失败输出，再去读相关源文件，最终总结失败原因。
2. **Lint / type check 建议**：Agent 运行 `uv run mypy src/` 或 `uv run ruff check .`，从输出中提炼下一步建议，但不做写操作。
3. **审批后 patch 并验证**：Agent 读取文件、提出 patch、等待审批、应用变更、重新运行测试并总结结果。
4. **验证失败并保留回滚路径**：Agent 在写后验证失败时，明确告诉用户备份位置、差异摘要和下一步处理建议。

## 建议实施顺序

为降低回归风险，建议按以下里程碑推进：

### M2.1 — 命令执行 MVP

- 完成 T18-T21
- 支持安全执行 `uv run pytest -q` / `uv run mypy src/`
- 不要求复杂恢复逻辑

### M2.2 — 命令执行收口

- 完成 T22-T25
- 补齐 verbose、审计、Reflector 行为和文档

### M3.1 — 审批写入 MVP

- 完成 T26-T31
- 只支持 `apply_patch`
- 支持暂停审批、批准恢复、diff 落盘

### M3.2 — 回滚与验证闭环

- 完成 T32-T35
- 支持备份、回滚、写后验证、文档收口

## 非目标与边界

以下能力不在本文件对应的阶段 2/3 首批实现范围内：

- 交互式 shell 会话
- 后台常驻进程管理
- 任意网络访问或联网安装依赖
- 跨 workspace 的命令执行或文件修改
- 对二进制文件的 patch / overwrite
- 绕过审批直接写文件

如后续需要这些能力，应单独开新阶段，而不是在阶段 2/3 中临时放宽策略。
