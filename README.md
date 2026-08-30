# EInkAgent 学习项目

目标：做一个面向电子墨水屏购买者的智能选购 Agent。

当前包含 SQLite 数据库基础和一个可复用的条件查询工具，不需要安装第三方依赖。

## 初始化并查看数据

```bash
python3 init_db.py
python3 seed_demo.py
python3 list_devices.py
```

运行后会在 `data/eink_devices.db` 生成数据库文件。

`seed_demo.py` 中的设备名称和参数全部是虚构的学习数据，不能作为真实购买依据。

## 条件查询工具

```bash
python3 -m examples.search_devices_demo
```

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
