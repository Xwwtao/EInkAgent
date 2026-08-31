# EInkAgent

![Tests](https://github.com/Xwwtao/EInkAgent/actions/workflows/tests.yml/badge.svg)

EInkAgent is a version-driven AI Agent portfolio project for helping users
choose E Ink devices from structured specifications and offer data.

The current v0.1 tool layer supports searching, inspecting, and comparing
devices through a tested SQLite repository.

> All current device and offer records are fictional demo data and must not be
> treated as real purchasing information.

## Current capabilities

- `search_devices(...)` filters devices by price, brand, screen size, weight,
  stylus support, color display, operating system, and category.
- `get_device_detail(device_id)` returns complete device details, returns `None`
  for an unknown device, and rejects invalid IDs.
- `compare_devices(device_ids)` preserves input order, removes duplicate IDs,
  and reports unknown devices.

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
```

The `/devices` endpoint supports the optional `brand`, `max_price`, and
`limit` query parameters. Invalid values return an HTTP 422 validation
response.

## 条件查询工具

```bash
python3 -m examples.search_devices_demo
```

## Architecture

```text
Examples / future API / future Agent
                 |
                 v
       device_repository.py
      search / detail / compare
                 |
                 v
            database.py
                 |
                 v
       SQLite: devices + offers
```

The repository layer keeps SQL and validation separate from future API and Agent interfaces, allowing those interfaces to reuse the same tested tools.

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
- [ ] v0.2 — FastAPI service
- [ ] v0.3 — Structured requirement parsing with an LLM
- [ ] v0.4 — Tool-calling Agent
- [ ] v0.5 — RAG with citations
- [ ] v0.6 — Memory and reliability
- [ ] v1.0 — Portfolio release

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
