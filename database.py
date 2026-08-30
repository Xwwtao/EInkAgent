from pathlib import Path
import sqlite3


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "eink_devices.db"
SCHEMA_PATH = PROJECT_DIR / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """创建数据库连接，并让查询结果可以通过字段名读取。"""
    DATA_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    """读取 schema.sql，并创建项目需要的表和索引。"""
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection() as connection:
        connection.executescript(schema)
