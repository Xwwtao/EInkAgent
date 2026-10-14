"""Report data quality for the imported real device and offer records.

The report is deliberately blunt about gaps: an unverified field is worse than
an empty one, so the report is meant to be read as a to-do list.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from database import get_connection

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = PROJECT_DIR / "data" / "data_quality_report.md"

# A price snapshot older than this is treated as stale.
OFFER_STALE_AFTER_DAYS = 30

# Devices created by seed_demo.py; they are demo data, not real records.
DEMO_BRANDS = ("DemoInk", "ColorLeaf")

PROVENANCE_COLUMNS = ("spec_source_url", "spec_verified_at")

SPEC_COLUMNS = (
    "screen_size_inches",
    "weight_g",
    "storage_gb",
    "mono_ppi",
    "operating_system",
    "release_year",
)


def _scalar(connection: Any, query: str, parameters: tuple[Any, ...] = ()) -> int:
    """Run a query that returns a single number."""
    row = connection.execute(query, parameters).fetchone()

    if row is None or row[0] is None:
        return 0

    return int(row[0])


def _total(connection: Any) -> int:
    return _scalar(connection, "SELECT COUNT(*) FROM devices")


def _demo_count(connection: Any) -> int:
    placeholders = ", ".join("?" for _ in DEMO_BRANDS)

    return _scalar(
        connection,
        f"SELECT COUNT(*) FROM devices WHERE brand IN ({placeholders})",
        DEMO_BRANDS,
    )


def _real_filter() -> tuple[str, tuple[str, ...]]:
    """Return a WHERE fragment that excludes the demo devices."""
    placeholders = ", ".join("?" for _ in DEMO_BRANDS)

    return f"brand NOT IN ({placeholders})", tuple(DEMO_BRANDS)


def _missing_count(connection: Any, column: str, real_only: bool = True) -> int:
    """Count devices whose column is NULL, ignoring demo data by default."""
    if not real_only:
        return _scalar(
            connection,
            f"SELECT COUNT(*) FROM devices WHERE {column} IS NULL",
        )

    where, parameters = _real_filter()

    return _scalar(
        connection,
        f"SELECT COUNT(*) FROM devices WHERE {column} IS NULL AND {where}",
        parameters,
    )


def _percent(part: int, whole: int) -> str:
    """Format a share of the real records as a percentage."""
    if whole == 0:
        return "0.0%"

    return f"{part / whole * 100:.1f}%"


def _stale_offers(connection: Any, today: date) -> int:
    cutoff = (today - timedelta(days=OFFER_STALE_AFTER_DAYS)).isoformat()

    where, parameters = _real_filter()

    return _scalar(
        connection,
        f"""
        SELECT COUNT(*) FROM offers
        JOIN devices ON devices.id = offers.device_id
        WHERE offers.collected_at < ? AND devices.{where}
        """,
        (cutoff, *parameters),
    )


def _devices_without_offers(connection: Any) -> int:
    where, parameters = _real_filter()

    return _scalar(
        connection,
        f"""
        SELECT COUNT(*) FROM devices
        WHERE id NOT IN (SELECT device_id FROM offers) AND {where}
        """,
        parameters,
    )


def build_report(today: date | None = None) -> str:
    """Build a Markdown data quality report."""
    report_day = today or date.today()

    with get_connection() as connection:
        total = _total(connection)
        demo = _demo_count(connection)
        real = total - demo

        where, parameters = _real_filter()

        offers_total = _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM offers
            JOIN devices ON devices.id = offers.device_id
            WHERE devices.{where}
            """,
            parameters,
        )
        devices_with_offers = _scalar(
            connection,
            f"""
            SELECT COUNT(DISTINCT offers.device_id) FROM offers
            JOIN devices ON devices.id = offers.device_id
            WHERE devices.{where}
            """,
            parameters,
        )

        lines: list[str] = [
            "# 数据质量报告",
            "",
            f"生成日期：{report_day.isoformat()}",
            "",
            "> 本报告只统计真实数据；演示数据（"
            + "、".join(DEMO_BRANDS)
            + "）单独计数。",
            "",
            "## 总量",
            "",
            f"- 设备：真实 {real} 台 / 演示 {demo} 台 / 合计 {total} 台",
            f"- 报价：{offers_total} 条，覆盖 {devices_with_offers} 台设备",
            "",
            "## 来源可追溯性",
            "",
            "| 字段 | 缺失数 | 缺失率 |",
            "| --- | --- | --- |",
        ]

        for column in PROVENANCE_COLUMNS:
            missing = _missing_count(connection, column)
            lines.append(
                f"| {column} | {missing} | {_percent(missing, real)} |"
            )

        no_provenance = _scalar(
            connection,
            f"""
            SELECT COUNT(*) FROM devices
            WHERE (spec_source_url IS NULL OR spec_verified_at IS NULL)
            AND {where}
            """,
            parameters,
        )
        lines.append("")
        lines.append(
            f"- 来源或核验时间缺失的真实设备：**{no_provenance}** 台"
            f"（{_percent(no_provenance, real)}）"
        )

        lines.extend(
            [
                "",
                "## 规格完整性",
                "",
                "| 字段 | 缺失数 | 缺失率 |",
                "| --- | --- | --- |",
            ]
        )

        for column in SPEC_COLUMNS:
            missing = _missing_count(connection, column)
            lines.append(
                f"| {column} | {missing} | {_percent(missing, real)} |"
            )

        lines.extend(["", "## 分类分布", ""])

        rows = connection.execute(
            f"""
            SELECT category, COUNT(*) AS count
            FROM devices
            WHERE {where}
            GROUP BY category
            ORDER BY category
            """,
            parameters,
        ).fetchall()

        if rows:
            for row in rows:
                lines.append(f"- {row['category']}：{row['count']} 台")
        else:
            lines.append("- 无数据")

        lines.extend(["", "## 报价时效", ""])
        lines.append(
            f"- 采集时间超过 {OFFER_STALE_AFTER_DAYS} 天的报价："
            f"{_stale_offers(connection, report_day)} 条"
        )

        devices_without_offers = _devices_without_offers(connection)
        lines.append(f"- 没有任何报价的真实设备：{devices_without_offers} 台")

        lines.extend(["", "## 达标检查", ""])
        lines.extend(
            [
                f"- [{'x' if real >= 20 else ' '}] 真实设备 ≥ 20 台（当前 {real}）",
                f"- [{'x' if offers_total >= 40 else ' '}] 报价 ≥ 40 条"
                f"（当前 {offers_total}）",
                f"- [{'x' if no_provenance == 0 and real > 0 else ' '}]"
                " 来源与核验时间 100% 非空",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the report to disk and print it."""
    report = build_report()
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"报告已写入：{DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
