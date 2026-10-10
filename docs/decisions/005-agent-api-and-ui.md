# 005 Agent HTTP 接口、多轮对话与网页界面（P3）

## 问题

1. Agent 只能通过命令行运行一次一问，没有可演示的入口。
2. `run_agent` 每次从零开始，追问"第二款多少钱"时模型不知道指什么。
3. 冒烟测试中模型把规格核验日期（`spec_verified_at`）说成了价格日期：
   工具只返回 `lowest_price_cny`，不返回这个价格来自哪个卖家、哪天采集。

## 决定

### 价格带来源

搜索与详情结果中的最低价改为取"最低价那一条报价"的完整信息：
`lowest_price_cny`、`price_seller`、`price_collected_at`、`price_url`。
用窗口函数 `ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY price_cny, collected_at DESC, id)`
取每台设备的第一条在售报价；同价时取最新采集的。
理由：模型需要的事实必须出现在工具结果里，否则它会用最接近的字段填补。

### 多轮对话：客户端保存历史（无状态服务）

| 方案 | 成本 | 取舍 |
| --- | --- | --- |
| **A. 界面保存历史，每次请求携带（选择）** | 只改接口参数 | 服务端不存状态，无需数据库；历史由客户端提供，可被伪造 |
| B. 服务端按会话 ID 存历史 | 需要存储与过期清理 | 属于 P5 记忆的范围，需要新增存储 |

- `MAX_HISTORY_MESSAGES = 10`（约 5 轮）：每多一条历史，token 与延迟都增加；
  选购对话通常在 5 轮内完成。超出时保留最近的消息。
- 历史只接受 `user` / `assistant` 角色的纯文本；不接受 `system` 和 `tool`，
  防止客户端伪造系统指令或工具结果。
- 不回传工具原始结果：模型之前的回答里有设备名，需要时可以用 `keyword` 重新查询。

### `POST /agent`

请求：`{"message": str (1–1000 字), "history": [{"role", "content"}] (≤ 20 条)}`
响应：`answer`、`status`（ok / degraded / blocked）、`tool_calls`（每次调用的名称、参数、
状态、错误类型、结果数、结果）、`latency_ms`、`model`。

- 模型客户端通过 FastAPI 依赖注入，测试中替换为假客户端，不发真实请求。
- 未配置 `DEEPSEEK_API_KEY` → 503，提示如何配置。
- 模型 API 错误：超时 → 504；其他 API 错误 → 502；响应中不包含异常原文。
  OpenAI SDK 设 `max_retries=2`，由 SDK 对超时、429、5xx 自动重试（可重试错误），
  400 / 401 等不可重试错误不重试。

### 网页界面

FastAPI 在 `GET /` 返回单个静态页面 `eink_agent/static/index.html`（原生 HTML/CSS/JS），
不引入前端框架、构建工具或新依赖。页面包含：聊天区、示例问题、
设备卡片（价格、卖家、采集日期、来源链接）、可展开的工具调用过程。

## 影响文件

`device_repository.py`、`eink_agent/agent.py`、`eink_agent/api.py`、
`eink_agent/static/index.html`（新增）、`tests/test_api.py`、`tests/test_agent.py`、
`tests/test_device_repository.py`、`.env.example`（新增，登记环境变量）

## 验证

`python -m pytest -q`；本地启动 `python -m uvicorn eink_agent.api:app` 后用浏览器完成
一次两轮对话（先推荐，再追问其中一款）。

## 已知限制

- 历史由客户端提供，服务端不校验其真实性；这不会带来越权，因为工具本身只读。
- 刷新页面后对话丢失（不使用浏览器存储，避免把对话留在共享电脑上）。
