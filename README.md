# EInkAgent

![Tests](https://github.com/Xwwtao/EInkAgent/actions/workflows/tests.yml/badge.svg)

EInkAgent is a version-driven AI Agent portfolio project for helping users
choose E Ink devices from structured specifications and offer data.

The current v0.3 release adds DeepSeek-powered natural-language requirement
parsing and a command-line search demo, alongside the existing FastAPI
device endpoints. An evaluation runner checks extraction against labeled cases.

> All current device and offer records are fictional demo data and must not be
> treated as real purchasing information.

## Current capabilities

- `search_devices(...)` filters devices by price, brand, screen size, weight,
  stylus support, color display, operating system, and category.
- `get_device_detail(device_id)` returns complete device details, returns `None`
  for an unknown device, and rejects invalid IDs.
- `compare_devices(device_ids)` preserves input order, removes duplicate IDs,
  and reports unknown devices.
- FastAPI exposes these tools through `GET /devices`,
  `GET /devices/{device_id}`, and `POST /devices/compare`.

## Quick start

These commands create `data/eink_devices.db`, initialize its schema, insert three fictional demo devices, and print the available records.

Requires Python 3.14.

```bash
git clone https://github.com/Xwwtao/EInkAgent.git
cd EInkAgent
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python init_db.py
python seed_demo.py
python list_devices.py
```

## HTTP API

Start the local API server:

```bash
python -m uvicorn eink_agent.api:app --reload
```

Open the interactive API documentation:

- Swagger UI: http://127.0.0.1:8000/docs

Example requests:

```bash
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/devices?brand=DemoInk&max_price=2000&limit=10"
curl "http://127.0.0.1:8000/devices/1"
curl -X POST "http://127.0.0.1:8000/devices/compare" \
  -H "Content-Type: application/json" \
  -d '{"device_ids": [2, 1, 2]}'
```

The `/devices` endpoint supports the optional `brand`, `max_price`, and
`limit` query parameters. Invalid values return an HTTP 422 validation
response.

The `GET /devices/{device_id}` endpoint returns one device. Unknown device IDs
return HTTP 404, while IDs smaller than 1 return HTTP 422.

The `POST /devices/compare` endpoint accepts two to five positive device IDs
in a JSON request body. It preserves the input order, removes duplicate IDs,
and returns HTTP 404 when a requested device does not exist.

## 条件查询工具

```bash
python3 -m examples.search_devices_demo
```

## Natural-language search
The demo uses DeepSeek JSON Output through the OpenAI Python SDK.
Pydantic validates the returned data before it is passed to `search_devices()`.

After completing Quick start, configure credentials in your current zsh terminal:

```bash
read -s "DEEPSEEK_API_KEY?DeepSeek API key: "
export DEEPSEEK_API_KEY
export DEEPSEEK_MODEL=deepseek-v4-flash
python -m examples.parse_requirements_demo
```

Example input:

```text
最多花两千元，重量不超过零点三公斤，要能用笔写
```

Expected search constraints:

```json
{
  "max_price": 2000,
  "max_weight_g": 300,
  "supports_stylus": true
}
```

The demo sends user input to DeepSeek and incurs API usage charges.
Never commit API keys. Credentials are read from environment variables;
the demo does not automatically load `.env` files.

### Validation and limitations

- Unit tests mock the model client and require no API key or network access.
- `evals/requirement_cases.json` stores four human-labeled cases.
  Live evaluation is run separately from pytest.
- A budget constraint was initially omitted; both cases passed once after
  a prompt revision. This is not an accuracy benchmark.
- JSON and Pydantic validation check structure and values, but cannot guarantee
  that every user requirement was understood.
- The workflow performs parsing followed by search; it does not yet implement
  autonomous tool selection.

## Requirement evaluation

Complete the DeepSeek environment-variable setup above, then run:

```bash
python -m examples.evaluate_requirements
```

This performs one API request per case and incurs API usage charges.
It is not run by default pytest or GitHub Actions.

The evaluator compares extracted constraints with human-labeled expectations:

- `missing`: an expected constraint was omitted
- `unexpected`: an unspecified constraint was added
- `incorrect`: a constraint has the wrong value
- `ERROR`: the request or output processing failed

Missing fields and null both mean unspecified. Explicit false and zero
remain constraints. A case passes only when all constraints match.
Errors remain in the total case count.

Reports are saved under `evals/runs/`, which Git ignores. Each report records
the start time, configured model name, prompt SHA-256 hash, expected and actual
constraints, and per-case outcomes.

A manual run on 2026-09-05 passed all four cases with `deepseek-v4-flash`.
This small set includes prompt-guided regression examples and is not an
independent accuracy benchmark.

Offline tests cover comparison behavior and recording failed requests:

```bash
python -m pytest
```

## Architecture

```text
Natural-language CLI → DeepSeek → DeviceRequirements
                  |
                  v
HTTP client → FastAPI → device_repository.py ← Python examples
                 |
                 v
            database.py
                 |
                 v
       SQLite: devices + offers
```

The CLI parses natural-language requirements before calling the repository.
FastAPI exposes search, detail, and comparison using explicit parameters;
natural-language parsing is not yet exposed through HTTP.

## 运行测试

安装测试依赖并运行测试：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```


## 文件说明

- `schema.sql`：定义数据库表和索引。
- `database.py`：连接并初始化 SQLite。
- `init_db.py`：创建数据库。
- `seed_demo.py`：插入虚构演示数据。
- `list_devices.py`：查看全部设备。
- `device_repository.py`：封装可供 API 和 Agent 调用的条件查询函数。
- `examples/search_devices_demo.py`：条件查询功能演示。

## 数据设计原则

- `devices` 保存相对稳定的设备规格。
- `offers` 保存会变化的价格、卖家和采集时间。
- 真实数据必须保存来源和核验时间。
- 不知道的参数使用 `NULL`，不能让模型猜测。

## Roadmap

- [x] v0.1 — SQLite data model and tested device tools
- [x] v0.2 — FastAPI service
- [x] v0.3 — Structured requirement parsing with an LLM
- [ ] v0.4 — Tool-calling Agent
- [ ] v0.5 — RAG with citations
- [ ] v0.6 — Memory and reliability
- [ ] v1.0 — Portfolio release

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
