# 第四阶段任务详细拆解：增强自主性与预算化执行

> 对应设计文档：`langgraph_linux_agent_design.md` § 10 阶段 4
>
> 适用基线：截至 2026-05-10 的当前仓库实现。阶段 1-3 已完成，默认 LLM 为 `deepseek-v4-pro`，通过 OpenAI 兼容接口接入 `https://api.deepseek.com`，并已具备原生 tool calling、受控命令执行、审批写入、写后验证与失败回滚链路。

## 文档目标

本文件把设计文档中的“阶段 4：增强自主性”进一步拆解为可实施、可测试、可验收的任务清单，作为后续增强 Agent 自主决策、预算约束、连续恢复与审批交互体验的详细设计基线。

**当前实现状态（2026-05-10）**：

- T18-T35 已完成。
- T36 已完成。
- T37 已完成。
- T38 已完成。
- T39 已完成。
- T40 已完成。
- T41 已完成。
- T42 已完成。
- T43-T44 未开始。

本文默认遵循以下原则：

- 继续沿用当前 `graph.py` 中的原生 tool calling，不引入新的结构化输出中间层。
- 优先扩展现有 `Planner -> PolicyGuard -> ToolExecutor -> Reflector -> Finalizer` 主路径，避免为了阶段 4 能力大幅改写图拓扑。
- 阶段 4 增强的是“有界自主性”，不是“更高权限”或“更多危险工具”。
- 所有新增自动恢复、计划修订、预算判定与审批交互行为，都必须进入现有审计日志和 verbose 输出。
- 预算约束必须是硬边界，而不是仅提供给模型参考的软提示。

## 阶段目标

### 阶段 4：增强自主性

目标是让 Agent 在显式预算和风险边界内，具备更稳定的计划更新能力、结构化反思评分、有限次连续错误恢复，以及更清晰的审批交互体验。

**阶段验收标准**：Agent 可以在有限预算内完成简单开发任务，在失败后进行有界恢复，并在预算耗尽或风险升高时给出明确、可审计、可中断的暂停或停止结果。

## 关键设计决策

### 1. 优先增强决策质量，而不是扩大工具面

- 阶段 4 不新增高权限工具。
- 核心工作是让现有读、搜、命令、写入与审批流程运行得更稳、更可控。
- “增强自主性”应表现为更好的计划更新、错误恢复和预算管理，而不是更激进的自动执行。

### 2. 计划更新与反思评分必须结构化落入状态

- 当前 `plan: list[str]` 足以表达阶段 1-3 的线性步骤，但不足以支撑阶段 4 的“计划更新”和“恢复评分”。
- 阶段 4 推荐在保持 `plan` 兼容的前提下，新增结构化 `plan_steps`、`plan_version`、`last_reflection` 等字段。
- 最终是否显示给用户，可以在 CLI / UI 层降级为简洁摘要，但 graph 内部必须保留结构化数据。

### 3. 连续错误恢复是“有界恢复”，不是无限自修复

- 每类失败必须带有恢复指纹和尝试计数。
- 对同一失败模式的自动恢复次数必须有限制。
- 当恢复收益持续下降、预算逼近耗尽或风险升级时，系统应停止自动恢复并转向总结/暂停。

### 4. 预算体系覆盖三个维度

- 现有 `max_iterations` 只覆盖工具轮次，不足以表达阶段 4 的真实运行成本。
- 阶段 4 至少需要补充：最大命令次数、最大运行时长、最大恢复次数。
- 预算状态应同时反馈给 Planner、Reflector、Finalizer 和审批 UI。

### 5. 审批 UI 先做“统一数据模型 + CLI 增强”，不强依赖 Web 前端

- 当前仓库主要入口仍是 CLI，因此阶段 4 的审批 UI 首先应强化 CLI 展示与恢复体验。
- 同时，应抽象出统一的审批展示 payload，为未来 HTTP / Web UI 复用留出接口。
- 阶段 4 的目标是“审批体验增强”，不是立即实现完整浏览器端前端系统。

## 任务总览

```text
阶段 4：增强自主性
T36 配置与状态扩展（预算 / 评分 / 恢复元数据）
T37 Planner 计划更新与步骤生命周期
T38 Reflector 反思评分与结构化决策
T39 连续错误恢复状态机
T40 任务预算执行与硬熔断
T41 审批交互模型与 CLI UI 增强
T42 审计与 verbose 可观测性扩展
T43 阶段 4 单元测试、集成测试、安全回归
T44 阶段 4 验收与文档收口
```

依赖关系：

- `T36 -> T37/T40 -> T38 -> T39 -> T41/T42 -> T43 -> T44`
- 推荐实现顺序：先补状态/预算基础，再补反思与恢复，最后做 UI 和审计增强。

---

## 阶段 4 任务拆解

### T36 — 配置与状态扩展（预算 / 评分 / 恢复元数据） 已完成

**文件**：`src/linux_agent/config.py`、`src/linux_agent/state.py`、必要时 `src/linux_agent/run_store.py`

**任务内容**：

- 在 `AgentConfig` 中新增阶段 4 所需配置，建议至少包括：
  - `max_command_count`
  - `max_runtime_seconds`
  - `max_plan_revisions`
  - `max_recovery_attempts_per_issue`
  - `budget_warning_ratio`
  - `reflection_replan_threshold`
  - `reflection_stop_threshold`
  - `approval_ui_mode`（如 `compact` / `detailed`）
- 在 `AgentState` 中新增结构化运行元数据，建议至少包括：
  - `started_at` 或等价的运行开始时间戳
  - `command_count`
  - `plan_version`
  - `plan_revision_count`
  - `plan_steps`
  - `last_reflection`
  - `recovery_state`
  - `budget_status`
  - `budget_stop_reason`
- `run_store.py` 需要支持这些新字段的持久化与恢复，确保暂停审批后恢复时预算状态不丢失。
- 所有字段必须提供保守默认值，不能破坏阶段 1-3 的已有运行路径。

**建议数据结构**：

```python
class PlanStep(TypedDict):
    id: str
    title: str
    status: Literal["pending", "in_progress", "completed", "blocked", "skipped"]
    rationale: str | None
    evidence_refs: list[int]


class ReflectionResult(TypedDict):
    score: int
    outcome: Literal["continue", "replan", "retry", "pause", "stop"]
    reason: str
    retryable: bool


class BudgetStatus(TypedDict):
    iteration_count: int
    command_count: int
    elapsed_seconds: int
    warning_triggered: bool
```

**验收标准**：

- `load_config()` 能在不破坏现有阶段 1-3 行为的前提下，返回完整阶段 4 配置。
- `AgentState` 可以表达计划修订、恢复尝试和预算状态。
- 暂停审批后的恢复运行不会丢失预算与恢复计数。

**实现结果（2026-05-10）**：

- `AgentConfig` 已新增阶段 4 所需预算、反思阈值和审批 UI 配置字段，并补充阈值一致性校验。
- `AgentState` 已新增 `started_at`、`command_count`、`plan_version`、`plan_revision_count`、`plan_steps`、`last_reflection`、`recovery_state`、`budget_status`、`budget_stop_reason`。
- `run_store.py` 已支持这些字段的持久化与恢复，并对旧快照缺失字段执行保守回填。
- 已新增 T36 回归测试，覆盖默认值、YAML 覆盖、legacy state 回填和新元数据 round-trip。

### T37 — Planner 计划更新与步骤生命周期 已完成

**文件**：`src/linux_agent/graph.py`、`src/linux_agent/state.py`

**任务内容**：

- 扩展 Planner prompt，使其显式理解：
  - 当前计划版本与步骤状态
  - 最近一次反思结论
  - 剩余预算
  - 是否正处于恢复流程
- 将当前线性的 `plan: list[str]` 升级为“可修订计划”，至少支持：
  - 新增步骤
  - 标记完成
  - 标记阻塞
  - 根据观察结果修订后续步骤
- 计划修订必须有理由，不能每轮无条件重写全部计划。
- 计划版本变更应被审计记录，避免最终只留下最新计划而看不到修订过程。
- 为兼容现有 CLI 展示，可保留扁平化的 `plan` 字段作为摘要层，而把结构化细节放入 `plan_steps`。

**建议实现约束**：

- 每次 Planner 最多修改一个有限范围的步骤集合，而不是整表重建。
- 当 `plan_revision_count` 达到上限时，Planner 只能继续执行或停止，不能无限重规划。
- 当前步骤与 `proposed_tool_call` 的语义必须一致，避免“计划说在读文件，实际却去跑命令”。

**验收标准**：

- Agent 可以在命令失败、验证失败、审批拒绝等场景下修订计划，而不是只在 `plan` 尾部堆叠新文本。
- 计划修订次数受限，超限时行为可预测。
- verbose 和 audit 中可以看到计划版本变化。

**实现结果（2026-05-10）**：

- Planner prompt 已接入结构化 `plan_steps`、`plan_version`、`plan_revision_count`、预算摘要、最近反思和恢复状态摘要。
- graph 已在每轮 Planner 后维护结构化步骤生命周期：初始化计划、失败后追加后续步骤、完成后标记 `completed`、失败切换后标记 `blocked`。
- `plan_update` 审计事件已扩展为包含 `plan_steps`、`plan_version`、`plan_revision_count`、`plan_revision_reason`，CLI verbose 也会显示这些字段。
- 已新增回归测试覆盖失败后计划修订和计划版本审计落盘。

### T38 — Reflector 反思评分与结构化决策 已完成

**文件**：`src/linux_agent/graph.py`，必要时新增 `src/linux_agent/reflection.py`

**任务内容**：

- 在 Reflector 中新增结构化反思输出，而不只是“是否继续”的布尔化判断。
- 反思评分建议至少考虑：
  - 最近一次 observation 是否产生新信息
  - 当前失败是否可恢复
  - 已消耗预算占比
  - 当前风险等级
  - 最近计划是否频繁抖动
- 输出建议至少包括：
  - `score`
  - `outcome`
  - `reason`
  - `retryable`
  - `recommended_next_action`
- 初版可以用确定性规则或启发式评分；是否引入第二个 LLM 调用做 reflection，可作为后续增强项，不应成为阶段 4 MVP 的前置条件。

**建议 outcome**：

- `continue`：继续按当前计划推进
- `replan`：要求 Planner 修订计划
- `retry`：允许一次恢复性重试
- `pause`：等待审批或人工介入
- `stop`：预算或恢复收益过低，停止运行

**验收标准**：

- Reflector 不再只靠少量硬编码条件决策，而能给出结构化评分和后续建议。
- 低分或高风险情况下，graph 会优先停止或重规划，而不是继续盲跑。
- 反思评分结果可进入 audit 和最终总结。

**实现结果（2026-05-10）**：

- Reflector 已新增确定性结构化反思结果，填充 `last_reflection.score / outcome / reason / retryable / recommended_next_action`。
- 评分已综合最近 observation 是否产生新信息、失败类型、预算压力、计划修订次数和当前恢复尝试次数。
- `reflector_action` 审计事件已新增 `reason=reflection_scored` 载荷，并在 `run_end` 中落入 `last_reflection`；最终回答也会在非 `continue` 场景追加反思摘要。
- 当前已实际使用的 outcome 包括 `continue`、`retry`、`replan`、`stop`；`pause` 仍保留在数据模型中，待后续审批交互增强阶段进一步接入。

### T39 — 连续错误恢复状态机 已完成

**文件**：`src/linux_agent/graph.py`，必要时新增 `src/linux_agent/recovery.py`

**任务内容**：

- 为常见失败场景建立有界恢复分类，建议至少覆盖：
  - 命令失败且输出中包含明确文件/行号
  - 搜索无结果
  - 文件不存在
  - 写后验证失败
  - 审批被拒绝后仍有低风险替代路径
- 为每类失败生成“恢复指纹”，避免系统在同一错误上重复兜圈。
- 恢复状态至少要跟踪：
  - 当前失败类型
  - 已尝试次数
  - 上次恢复动作
  - 是否仍可恢复
- 恢复策略建议采用“有限动作模板”，例如：
  - 先读相关文件，再决定是否重试命令
  - 搜索候选路径，而不是重复读同一个不存在文件
  - 写后验证失败时，优先读错误上下文，再决定是否再修一轮
- 当相同恢复指纹超过上限时，系统必须停止自动恢复并总结现状。

**验收标准**：

- Agent 可以在简单连续错误场景下进行 1-2 轮有意义的恢复，而不是立即失败或无限重试。
- 相同错误不会无限重复尝试。
- 恢复计数和恢复原因会被审计记录。

**实现结果（2026-05-10）**：

- graph 已实现失败分类与恢复指纹，当前覆盖 `command_failure`、`command_timeout`、`search_no_results`、`file_missing` / `file_read_error`、`path_missing` / `path_access_error`、`verification_failed`。
- `recovery_state` 现在会跟踪 `issue_type`、`fingerprint`、`attempt_count`、`last_action`、`can_retry`，并在恢复成功后自动清空。
- 对同一恢复指纹的重复失败会递增尝试次数；超过 `max_recovery_attempts_per_issue` 后，Reflector 会停止自动恢复并通过 `budget_stop_reason=max_recovery_attempts` 结束运行。
- `reflector_action` 审计事件已新增 `recovery_attempted`、`recovery_exhausted`、`recovery_cleared` 语义载荷；已新增回归测试覆盖首次恢复、恢复成功清空状态和重复失败熔断。

### T40 — 任务预算执行与硬熔断 已完成

**文件**：`src/linux_agent/config.py`、`src/linux_agent/graph.py`、`src/linux_agent/app.py`

**任务内容**：

- 在现有 `max_iterations` 基础上，新增并执行：
  - 最大命令次数限制
  - 最大运行时长限制
  - 最大计划修订次数
  - 最大恢复尝试次数
- Planner prompt 中要显示剩余预算，让模型具备预算感知。
- Reflector / Finalizer 要能区分不同预算耗尽原因，例如：
  - `max_iterations`
  - `max_command_count`
  - `max_runtime_seconds`
  - `max_recovery_attempts`
- CLI 在 verbose 模式下应能显示预算消耗和预算警告。
- 预算检查建议作为 graph 内共享 helper，而不是立即新增一个专用新节点，以减少对现有拓扑的扰动。

**建议行为**：

- 预算接近阈值时先发 `budget_warning`，不是等完全耗尽才中断。
- 预算耗尽时进入 Finalizer，总结：
  - 已完成的步骤
  - 当前阻塞点
  - 若有恢复/回滚路径，应该一并给出

**验收标准**：

- Agent 不会在命令数量、运行时长或恢复次数上无限扩张。
- 不同预算耗尽原因能在 CLI、audit 和最终回答中区分显示。
- 暂停审批后恢复运行时，预算消耗延续原 run，而不是被重置。

**实现结果（2026-05-10）**：

- `graph.py` 已把 `max_command_count`、`max_runtime_seconds`、`max_plan_revisions`、`max_recovery_attempts_per_issue` 接入实际运行预算。
- `tool_executor` 会维护 `command_count` 和 `budget_status`；Planner/Reflector 会在预算耗尽时设置 `budget_stop_reason` 并硬停止后续执行。
- Planner prompt 已显示剩余预算；run_end 审计事件已包含 `budget_status`、`budget_remaining`、`budget_stop_reason`。
- CLI verbose 已显示预算快照、剩余预算和预算停止原因；新增测试覆盖命令预算熔断、运行时预算熔断和 verbose 渲染。

### T41 — 审批交互模型与 CLI UI 增强 已完成

**文件**：`src/linux_agent/app.py`、必要时新增 `src/linux_agent/approval_ui.py`、`src/linux_agent/run_store.py`

**任务内容**：

- 将当前审批输出从“能恢复”提升为“便于判断是否恢复”。
- 统一审批展示 payload，建议至少包含：
  - 工具调用摘要
  - 影响文件
  - diff 预览
  - 备份 / 回滚路径
  - 建议验证命令
  - 当前剩余预算
  - 当前恢复状态
- CLI 至少应增强：
  - 暂停时的审批卡片式输出
  - 重新查看某个 pending run 的审批详情（如 `--show-pending-run <run_id>`）
  - 审批响应原因记录（如拒绝原因、批准备注）
- 为未来 HTTP / Web UI 留出通用数据结构，不要求阶段 4 立即实现完整前端页面。

**建议边界**：

- 阶段 4 不要求浏览器富前端，但要求 CLI 具备足够清晰的交互信息。
- 审批 UI 不能泄露被策略裁剪掉的敏感内容；diff 预览仍应受大小限制。

**验收标准**：

- 用户在 CLI 中能清楚看到风险、影响、回滚和预算信息，再决定是否批准。
- 审批展示 payload 可被未来 API / UI 复用，而不是硬编码在 stderr 文案里。
- 恢复和拒绝操作都可带结构化说明并进入 audit。

**实现结果（2026-05-10）**：

- 新增 `src/linux_agent/approval_ui.py`，将审批展示抽象成可复用 `approval view` 载荷，统一包含工具摘要、影响文件、diff 预览、备份/回滚命令、建议验证命令、预算剩余、恢复状态、计划步骤与恢复命令。
- `app.py` 已新增 `--show-pending-run <run_id>`，可在不恢复执行的情况下重新查看待审批 run 的完整审批卡片；暂停时 CLI 也会直接打印同一套审批视图。
- `--resume-run ... --approve/--reject` 现已支持 `--decision-note "..."`，批准备注或拒绝原因会通过 graph 传入审批恢复链路并在拒绝场景同步到最终回答。
- `policy.py` 生成的 `ApprovalRequest` 已补充 `affected_files`、`risk_level`、`suggested_verification_command`、`rollback_command`，为未来 HTTP / Web UI 复用保留了稳定字段。

### T42 — 审计与 verbose 可观测性扩展 已完成

**文件**：`src/linux_agent/audit.py`、`src/linux_agent/app.py`、`src/linux_agent/graph.py`

**任务内容**：

- 新增阶段 4 关键审计事件，建议至少包括：
  - `plan_revised`
  - `reflection_scored`
  - `recovery_attempted`
  - `recovery_exhausted`
  - `budget_warning`
  - `budget_exhausted`
  - `approval_presented`
  - `approval_response`
- `run_end` 应补充：
  - 计划修订次数
  - 恢复次数
  - 命令预算使用情况
  - 运行时长
  - 最终预算停止原因（如有）
- verbose 输出建议支持：
  - 当前 reflection score
  - 当前 outcome
  - 剩余预算摘要
  - 当前恢复指纹和恢复次数

**验收标准**：

- 阶段 4 的关键决策链路可以在 JSONL 中完整回放。
- `--verbose` 下，用户能看清“为什么继续/为什么停止/为什么重规划”。
- 日志字段足以支持未来阶段的 run replay 或简单可视化。

**实现结果（2026-05-10）**：

- `audit.py` 已新增 `plan_revised`、`reflection_scored`、`recovery_attempted`、`recovery_exhausted`、`recovery_cleared`、`budget_warning`、`budget_exhausted`、`approval_presented`、`approval_response` 等阶段 4 独立事件；同时保留原有 `reflector_action` 作为兼容层。
- `graph.py` 已在 planner / approval_pause / resume_gate / reflector 中实际发出上述事件，并为 `approval_presented` 写入完整审批视图，为 `approval_response` 写入用户动作与备注。
- `run_end` 已补充 `runtime_seconds`、`recovery_attempt_total`、`budget_usage`、`budget_stop_reason`、`last_reflection`、`recovery_state` 等字段，阶段 4 的关键决策链路现在可直接从 JSONL 回放。
- `app.py` verbose printer 已支持独立渲染计划修订、反思评分、恢复尝试、预算警告/耗尽、审批展示与审批响应，控制台可直接看到 reflection score、outcome、剩余预算与恢复指纹/次数。

### T43 — 阶段 4 单元测试、集成测试、安全回归

**文件**：`tests/` 下新增或扩展相关测试

**任务内容**：

- 单元测试覆盖：
  - 新预算字段默认值与 YAML 覆盖
  - 计划步骤状态流转
  - reflection score 计算与阈值分支
  - recovery 指纹与恢复次数上限
  - 审批展示 payload 的裁剪与字段完整性
- 集成测试覆盖：
  - 命令失败 -> 读上下文 -> 恢复 -> 再次验证
  - 写后验证失败 -> 再修一轮或预算耗尽停止
  - 命令预算耗尽后安全停止
  - 运行时长预算耗尽后安全停止
  - 审批 UI 展示 pending run 详情
- 安全回归覆盖：
  - 预算警告或审批 UI 不能泄露被裁剪的敏感内容
  - 连续恢复不能绕过 policy 或预算限制
  - 相同失败指纹不会触发无限循环

**验收标准**：

- 阶段 4 的核心风险点和关键决策路径都有自动化测试。
- 阶段 1-3 全部回归保持绿色。
- 自动恢复、预算停止和审批展示三条链路都有端到端覆盖。

### T44 — 阶段 4 验收与文档收口

**文件**：`README.md`、`config.yaml`、`docs/langgraph_linux_agent_design.md`、必要时新增演练文档

**任务内容**：

- 更新 README，说明：
  - 预算配置怎么设
  - 命令/运行时长耗尽时会发生什么
  - 审批 UI 增强后，用户如何查看 pending run 详情
- 更新设计文档，标明阶段 4 已实现的增强自主性范围与边界。
- 补充至少 2 个手工验收流程：
  - 简单开发任务在预算内完成
  - 连续错误恢复后停止并总结
- 说明哪些能力仍不在阶段 4 范围内，例如：
  - 无界自动修复
  - 多 Agent 自协作
  - 完整 Web UI 产品化

**验收标准**：

- 用户能根据文档理解阶段 4 的预算、恢复和审批体验。
- 文档能明确地区分“当前已实现能力”和“未来扩展方向”。

---

## 跨阶段验收场景建议

建议至少保留以下 4 个标准场景作为阶段 4 里程碑：

1. **预算内完成任务**：Agent 读取上下文、运行测试、提出修复、审批写入、验证通过，并在命令预算内完成任务。
2. **连续错误恢复**：Agent 先遇到命令失败，再读取相关文件做一次恢复，第二次仍失败后在恢复次数上限处安全停止。
3. **预算耗尽安全停止**：Agent 在命令预算或运行时长预算逼近时收到 warning，并在耗尽后给出当前进度与阻塞摘要。
4. **审批增强体验**：用户可以查看 pending run 的风险、diff、回滚和预算信息，再决定批准或拒绝。

## 建议实施顺序

为降低回归风险，建议按以下里程碑推进：

### M4.1 — 预算与计划基线

- 完成 T36-T37
- 建立结构化计划、预算字段与运行状态持久化
- 不要求复杂自动恢复

### M4.2 — 反思评分与连续恢复 MVP

- 完成 T38-T40
- 支持 reflection score、有限恢复、命令/运行时长预算熔断

### M4.3 — 审批交互与可观测性增强

- 完成 T41-T42
- CLI 能清楚展示审批、预算和恢复状态
- audit/verbose 能解释主要决策原因

### M4.4 — 测试与收口

- 完成 T43-T44
- 形成完整自动化回归和文档验收流程

## 非目标与边界

以下能力不在本文件对应的阶段 4 首批实现范围内：

- 多 Agent 分工与协作
- 自动安装依赖或联网修复供应链问题
- 无上限自动重试或长时间后台自治运行
- 完整 Web 审批产品化界面
- 通过预算或恢复逻辑绕过现有 policy / approval 边界

如后续需要这些能力，应在阶段 5 或更后续阶段单独规划，而不是在阶段 4 中临时放宽边界。