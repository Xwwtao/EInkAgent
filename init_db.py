from database import DB_PATH, initialize_database


if __name__ == "__main__":
    initialize_database()
    print(f"数据库初始化完成：{DB_PATH}")
