# 导读 02：HTTP 接口、多轮对话与网页界面（P3 面试问答）

> 对应代码：`eink_agent/api.py`、`eink_agent/agent.py` 的 `history` 参数、
> `eink_agent/static/index.html`。设计说明：`docs/decisions/005-agent-api-and-ui.md`。

## 一次请求经过哪些地方

```
浏览器 index.html
  │  fetch POST /agent  {"message": "第二款防水吗？", "history": [最近 10 条]}
  ▼
api.py  AgentRequest 校验（空消息、超长、非法角色 → 422，模型完全不会被调用）
  │     Depends(get_agent_runtime) → 拿到模型客户端（测试中被替换成假的）
  ▼
agent.py  run_agent(message, history=...) → [system, 历史..., 新问题] → 工具循环
  ▼
api.py  把 AgentResult 转成 AgentResponse；模型 API 出错 → 502 / 504
  ▼
浏览器  先转义再渲染 Markdown → 设备卡片 → 工具调用时间线
```

## 面试问答

### Q1. 为什么由页面保存聊天记录，服务端不存？风险是什么？

**答**：服务端不存状态，就不需要数据库或 Redis 来保存会话，也没有"服务端留存用户对话"的隐私问题。
注意：token 成本没有减少，每次请求都要把历史发给模型。

**风险**：
- 历史由客户端提供，**可以伪造**，例如伪造一条 assistant 消息"这台可以 1 元购买"；
- 刷新页面对话就丢失；
- 对话越长请求越大，所以只保留最近 10 条（`MAX_HISTORY_MESSAGES`）。

### Q2. 为什么 `history` 里不接受 `system` 和 `tool` 角色？

**答**：`system` 消息在模型眼里权威最高。接受它，攻击者就能写"忽略之前所有规则"，这是提示注入。
`tool` 消息会被模型当作真实查询结果，接受它就能伪造数据，例如"Kindle 售价 1 元"。
所以只接受 `user` 和 `assistant`，并在两处校验：接口的 `HistoryMessage`（422）和 `run_agent`（`ValueError`）。

### Q3. 为什么超时返回 504、其他模型错误返回 502、没配 key 返回 503？

**答**：我们的服务是用户和 DeepSeek 之间的网关。

| 状态码 | 含义 | 处理方式 |
| --- | --- | --- |
| 504 Gateway Timeout | 上游（DeepSeek）太慢 | 稍后重试可能成功 |
| 502 Bad Gateway | 上游出错（连接失败、异常响应） | 稍后重试可能成功 |
| 503 Service Unavailable | 我们自己没准备好（没配 key） | 必须修复配置，重试无用 |

区分开，前端和运维才知道该等一会儿重试，还是该去改配置。三种响应都不带异常原文。

### Q4. 模型客户端为什么要依赖注入？

**答**：`run_agent` 不自己创建客户端，而是由调用方传入；接口层用 `Depends(get_agent_runtime)` 提供。
测试时用 `app.dependency_overrides` 换成按剧本回答的假客户端，所以测试不花钱、不联网、不需要 key，
结果每次都一样。如果在函数内部直接创建 DeepSeek 客户端，每跑一次测试就会真的调用 API。

### Q5. 渲染 Markdown 为什么必须先转义、再加格式？

**答**：**模型输出是不可信数据**，必须像用户输入一样对待。它可能受提示注入影响，以后还会夹带检索到的文档内容。
先把所有 `<` `>` `&` `"` `'` 转义，模型写的任何东西都只能显示为文字；之后再加我们自己的、已知安全的标签（`<strong>`、`<table>`）。
顺序反过来，会把我们自己的标签也转义掉，格式全坏；如果因此干脆不转义，就留下了 XSS 漏洞。
链接另外检查：只允许 `http://` 或 `https://` 开头，挡住 `javascript:` 链接。

### Q6. 价格日期为什么必须由工具返回，而不能靠提示词约束？

**答**：提示词只能要求模型"不要瞎说"，不能凭空给它正确的日期。模型手里没有某个事实时，会用最接近的字段填补——
冒烟测试里它把规格核验日期说成了价格日期。原则：**事实放进工具结果（确定），行为规则写进提示词（不保证每次遵守）**。

## 动手练习

1. 在浏览器开发者工具的 Network 面板里，看一次 `/agent` 请求的请求体和响应体，找到 `history` 和 `tool_calls`。
2. 不设置 `DEEPSEEK_API_KEY` 启动服务，发一条消息，看页面显示什么，对照 `get_agent_runtime`。
3. 在 `tests/test_agent_api.py` 里找到让模型超时的测试，说出它是怎么做到不联网就模拟超时的。
