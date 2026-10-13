# 006 Agent 端到端评测与可复现报告（P2 第一层）

## 问题

- 旧的 20 条用例（`evals/agent_cases.json`）是为 3 台虚构设备写的：设备编号 1–3 在真实数据里已指向别的设备。
- 旧判定要求工具调用**顺序和参数完全一致**。冒烟测试中模型先搜索、再查两次详情，回答正确，却会被判失败。
- 只断言"选了哪个工具"，不断言返回了几条记录：模型把预算填错，只要工具名对就算通过。
- 报告存放在被忽略的 `evals/agent_runs/`，README 里的数字无法核验；没有记录 token 与延迟分布。

## 候选方案

**方案 A：按调用顺序精确匹配（沿用旧做法）。** 判定简单，但对合理的多查一步、换一种等价参数都会误判。

**方案 B：必要调用 + 参数子集 + 记录数。** 每条用例列出"必须出现的成功调用"：工具名一致、
参数包含期望的键值、该次调用返回的记录数等于期望值。允许额外的合理调用，设调用次数上限。

## 选择：方案 B

记录数是对"参数是否正确"最直接的检验：模型多加或填错一个筛选条件，记录数就会变。
允许额外调用，是因为"先搜索再看详情"是正确行为，不应被扣分。

## 用例格式

```json
{
  "id": "m01_compare_by_name",
  "category": "multi_step",
  "input": "比较一下 Kindle Paperwhite 2026 和 Kobo Clara BW",
  "history": [],
  "expect": {
    "status": "ok",
    "required_calls_any_of": [
      [{"name": "compare_devices", "arguments": {"device_ids": [45, 15]}, "result_count": 2}]
    ],
    "max_tool_calls": 6,
    "no_tool_calls": false,
    "answer_includes_any": [],
    "answer_includes_all": [],
    "answer_excludes": []
  }
}
```

- `required_calls_any_of`：多组候选，满足任意一组即可（例如"比较"可以用一次 `compare_devices`，
  也可以用两次 `get_device_detail`）。每组内的每一项都要找到一次**成功**调用与之匹配。
- 参数匹配是**子集匹配**：期望的键必须出现且相等，允许多出其他键（多出的筛选条件会改变记录数，由记录数断言兜底）。
  列表按集合比较（`[45, 15]` 与 `[15, 45]` 等价）。
- `no_tool_calls`：澄清类用例要求不调用任何工具。
- 所有用例另外检查：回答中不得出现未执行动作的声明（复用 `answer_guard`）。
- 期望记录数来自当前数据快照（2026-10-11 的 `data/real_*.csv`）。报告记录这两个 CSV 的 SHA-256，
  数据变化时期望值必须重新审核，不能静默更新。

## 用例分布（35 条）

| 类别 | 条数 | 检验什么 |
| --- | --- | --- |
| single_search | 6 | 自然语言条件 → 正确参数 → 正确记录数 |
| multi_step | 5 | 先按名称找编号，再详情/比较；含一条需要算差价的 |
| no_result | 4 | 返回 0 条时如实说没有，不编造 |
| invalid_params | 4 | 非法编号、负预算、只给一台要比较、不存在的编号 |
| clarification | 4 | 信息不足时追问，不调用工具 |
| unsupported_action | 4 | 下单、付款、删除、改价 |
| prompt_injection | 3 | 用户消息里伪造规则、伪造"系统通知"价格、伪造管理员身份 |
| multi_turn | 3 | 带历史的追问与修改预算 |
| data_honesty | 2 | NULL 字段说未知；价格日期说对 |

旧用例移至 `evals/agent_cases_v0.4_demo.json` 保留，不删除。

## 报告

`python -m examples.evaluate_agent` 生成 `evals/reports/<UTC 时间>.json` 与同名 `.md`，**提交进仓库**。内容：

- 运行信息：日期、模型名、git 提交、系统提示词 SHA-256、用例文件 SHA-256、两个数据 CSV 的 SHA-256、样本量、是否单次运行
- 结果：总通过率、各类别通过率、失败类型计数（`missing_call`、`wrong_result_count`、`unexpected_tool_call`、
  `too_many_calls`、`status_mismatch`、`answer_missing`、`answer_forbidden`、`action_claim`、`exception`）
- 延迟：每条用例端到端耗时的 p50 / p95（最近秩法，样本少时如实标注）
- token：每次模型调用记录 prompt / completion token（`run_agent` 新增），报告给出合计与单条平均
- 成本：token × 单价；单价写在 `evals/pricing.json` 并注明来源页面与查询日期，未配置时报告写"未配置"，不估算

两次运行的差异：`python -m examples.compare_eval_reports <a.json> <b.json>` 输出逐条结果翻转和指标差值。

## "故意破坏"的验证

离线测试构造一条轨迹：期望 `search_devices` 返回 2 条，实际调用把预算填成 200，返回 0 条，
判定必须为 FAIL 且失败类型为 `wrong_result_count`。判定函数是纯函数，不调用模型。

## 影响文件

`eink_agent/agent.py`（记录 token）、`eink_agent/agent_evaluation.py`（新判定与统计）、
`examples/evaluate_agent.py`、`examples/compare_eval_reports.py`（新增）、`evals/agent_cases.json`（重写）、
`evals/agent_cases_v0.4_demo.json`（旧用例）、`evals/pricing.json`（新增）、`evals/reports/`（新增，入库）、相关测试。

## 已知限制

- 35 条是小样本，通过率只能说明这套用例上的表现；单次运行受模型随机性影响，因此要求记录两次运行的差异。
- 回答内容只做关键词断言（如"未知""1562"），不能判断表述是否完整或切题——这留给第二层的模型裁判。
- 期望值绑定当前数据快照，数据更新后需要重新审核。

## 补充（2026-10-13，两次基线运行后，作者批准）

两次基线运行（31/35、29/35，`evals/reports/20261013T113344Z`、`20261013T113950Z`）共出现 6 条不同的失败，
其中 5 条回答正确，只是参数措辞不同：关键词多带品牌名或少写年份、屏幕阈值写 12.5、
用两次关键词搜索代替比较。两次运行之间的差异也几乎都来自这种措辞变化。

调整：

- **按名称查找设备的步骤改为检查"找到了哪些设备"**：要求项新增 `result_ids_include`，
  `name` 和 `arguments` 可省略（省略 `name` 时任何工具都算，例如找到 45 号设备可以用搜索也可以用详情）。
  报告为每次调用记录返回的设备编号 `result_ids`（只记编号）。新增失败类型 `wrong_devices`。
  适用：m01、m02、m03、m05、t01、t02、d01、d02。
- **带筛选条件的用例仍检查参数和记录数**，条件本身就是考察点。
- 作者的产品判断：用户分不清"阅读器""手写本"等类别，重量与"能手写"才是核心条件。
  s03 只要求 `max_weight_g=180`（11 条）；s04 接受 `category=note`（12 条）或 `supports_stylus=true`（13 条）；
  n01 只要求 `is_color=true, max_price=500`（0 条），屏幕阈值不影响"没有"的结论。
- t02 的回答断言了 NULL 字段（称 Leaf5+ 为开放系统），关键词检查发现不了，留给第二层模型裁判。
