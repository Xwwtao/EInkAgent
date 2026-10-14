"""Tests for session memory storage (decision 009)."""

import sqlite3

import pytest

from eink_agent.memory import SQLiteMemoryStore, describe_preferences


@pytest.fixture
def store(tmp_path):
    return SQLiteMemoryStore(tmp_path / "memory.db")


def test_new_session_has_no_memory(store):
    assert store.messages("s1") == []
    assert store.preferences("s1") == {}
    assert store.preference_history("s1") == []
    assert store.summary("s1") == ("", 0)


def test_messages_are_kept_in_order_per_session(store):
    store.add_messages("s1", [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "请问预算？"}])
    store.add_messages("s2", [{"role": "user", "content": "别的会话"}])
    store.add_messages("s1", [{"role": "user", "content": "2000"}])

    assert [m["content"] for m in store.messages("s1")] == ["你好", "请问预算？", "2000"]
    assert [m["content"] for m in store.messages("s2")] == ["别的会话"]


def test_newer_preference_overrides_older_and_history_is_kept(store):
    store.set_preferences("s1", {"max_price": 2000, "supports_stylus": True})
    store.set_preferences("s1", {"max_price": 1500})

    assert store.preferences("s1") == {"max_price": 1500, "supports_stylus": True}
    history = store.preference_history("s1")
    assert [(item["field"], item["value"]) for item in history] == [
        ("max_price", 2000),
        ("supports_stylus", True),
        ("max_price", 1500),
    ]
    assert all(item["created_at"] for item in history)


def test_null_withdraws_a_preference_but_stays_in_history(store):
    store.set_preferences("s1", {"supports_stylus": True})
    store.set_preferences("s1", {"supports_stylus": None})

    assert store.preferences("s1") == {}
    assert [item["value"] for item in store.preference_history("s1")] == [True, None]


def test_preferences_do_not_leak_between_sessions(store):
    store.set_preferences("s1", {"max_price": 2000})

    assert store.preferences("s2") == {}


def test_summary_is_replaced_not_appended(store):
    store.save_summary("s1", "用户想要手写。", 6)
    store.save_summary("s1", "用户想要手写，预算 1500。", 12)

    assert store.summary("s1") == ("用户想要手写，预算 1500。", 12)


def test_memory_survives_reopening_the_file(tmp_path):
    SQLiteMemoryStore(tmp_path / "memory.db").set_preferences("s1", {"category": "note"})

    assert SQLiteMemoryStore(tmp_path / "memory.db").preferences("s1") == {"category": "note"}


def test_unreadable_file_raises_instead_of_returning_empty(tmp_path):
    # The caller decides how to degrade; the store must not hide the failure.
    path = tmp_path / "memory.db"
    path.write_text("not a database", encoding="utf-8")

    with pytest.raises(sqlite3.DatabaseError):
        SQLiteMemoryStore(path).preferences("s1")


def test_preferences_are_described_in_plain_chinese():
    text = describe_preferences(
        {"max_price": 1500, "supports_stylus": True, "is_color": False, "category": "note", "max_weight_g": 300}
    )

    assert "1500" in text and "手写" in text and "黑白" in text and "300" in text
    assert "max_price" not in text and "True" not in text


def test_no_preferences_describe_as_empty():
    assert describe_preferences({}) == ""
