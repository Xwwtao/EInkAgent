# 010 MCP 服务与配套 Skill

## 问题

设备工具只能被本项目自己的 Agent 使用。Claude Desktop、Claude Code 等支持 MCP 的客户端想查同一个设备库，
需要一个标准接口；同时这些客户端的模型并不知道本项目已经验证过的使用规则
（未知字段不能推断、价格要带卖家和日期、按名称查询时不套用预算）。

## 候选方案（工具怎么暴露）

**A：SDK 的高层 `MCPServer`，用装饰器把每个工具重写成函数。** 写法最短，但参数结构从函数签名生成，
等于把 `DeviceRequirements` 等参数模型再写一遍，两份定义会慢慢不一致。

**B：SDK 的底层 `Server`，工具列表直接取 `offered_tools()`，执行走 Agent 用的同一个函数。**
多写十几行处理请求的代码，但工具名、说明、参数结构、校验和错误格式只有一份。

**选 B。** 依赖：官方 Python SDK `mcp==2.3.0`（作者 2026-10-15 同意新增）。

## 设计

- `eink_agent/mcp_server.py`：`build_server(knowledge=None)` 返回服务对象；`python -m eink_agent.mcp_server` 以 stdio 运行。
- 暴露 `search_devices`、`get_device_detail`、`compare_devices`；配置了向量服务时再加 `search_knowledge`。
  **不暴露 `remember_preferences`**：它依赖本项目网页的会话，MCP 客户端有自己的上下文。
- 每个工具标注只读（`read_only_hint`），不会修改任何数据。
- 参数校验、未知工具、工具内部异常都转成与 Agent 相同的结构化错误，以 `is_error` 结果返回，不回传异常原文。
  为此把 `agent.py` 中执行单个工具并转换错误的函数公开为 `run_tool`，Agent 与 MCP 共用。
- 服务的 `instructions` 写明核心规则（只读、未知字段不推断、价格带来源）。
- stdio 模式下标准输出是协议通道，服务代码不得 `print`。

## Skill

`skills/eink-shopping/SKILL.md`：给使用这个 MCP 服务的 Claude 一份做法说明——什么时候先追问、怎么组合工具、
怎么写价格来源、知识检索为空时怎么说。内容来自本项目系统提示词中已在评测里验证过的规则。
MCP 提供能力，Skill 提供做法；Skill 只引用服务中真实存在的工具名，由测试检查。

## 验证

- 离线测试用 SDK 的进程内客户端连接服务：工具列表与 Agent 一致且不含记忆工具；正常调用返回数据；
  非法参数、未知工具、工具异常返回结构化错误且不含异常原文；Skill 引用的工具名都存在。
- 手动验证：用 SDK 的 stdio 客户端启动 `python -m eink_agent.mcp_server` 子进程，列出工具并调用一次搜索。

## 影响文件

`eink_agent/mcp_server.py`（新增）、`eink_agent/agent.py`（公开 `run_tool`）、`skills/eink-shopping/SKILL.md`（新增）、
`requirements.txt`、`tests/test_mcp_server.py`（新增）、README。

## 已知限制

- 只做 stdio（本机客户端），不做远程 HTTP 部署与鉴权。
- MCP 客户端里的模型不受本项目答案校验（`answer_guard`）约束；Skill 只是说明，不能强制执行。
- 没有对"Claude 通过 MCP 使用这些工具"做评测，不对其效果下结论。
