
# 002 设备标志位使用 NULL 表示未知

## 问题

原 schema 使用 NOT NULL DEFAULT 0，无法区分“明确不支持”和“来源未公布”。

## 候选方案

方案 A：允许 NULL，使用 NULL / 0 / 1。
方案 B：保留非空列，引入 2 表示 unknown。

## 选择

选择方案 A。NULL 是 SQLite 原生的未知语义，现有等值筛选会自动排除未知记录，不需要增加特殊枚举和转换逻辑。

## 行为定义

- 1：来源明确说明支持
- 0：来源明确说明不支持
- NULL：来源未公布或无法确认
- offers.is_available 仍然必须是 0 或 1，不允许 NULL

## 影响文件

- schema.sql
- data_import.py
- tests/test_data_import.py

## 验证

运行 python -m pytest -q。
验证未知设备标志位保存为 NULL，且不会被 is_open_system=False 查询命中。
