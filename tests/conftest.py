"""shared pytest fixtures for isolated database test."""

import pytest

import database
from seed_demo import seed_demo_data

@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch) -> None:
    """Give every test a freshly seeded temporary SQLite database."""
    temporary_db_path = tmp_path / "eink_devices.db"
    monkeypatch.setattr(database, "DB_PATH", temporary_db_path)
    seed_demo_data()