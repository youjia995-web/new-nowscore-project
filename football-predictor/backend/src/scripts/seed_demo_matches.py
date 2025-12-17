#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插入一批演示用比赛到 SQLite 的 matches_manual 表，以便参数标定脚本有样本。
"""
import sqlite3
import os
from datetime import datetime, timedelta

# 直接使用 settings.db_path 的绝对路径以避免相对路径问题
from pathlib import Path
from importlib import import_module

# 读取配置（无需包结构，直接定位到 backend/src）
BASE = Path(__file__).resolve().parents[1]  # 到 backend/src
CONFIG_PATH = BASE / "config.py"

import sys
sys.path.append(str(BASE))
settings = import_module("config").settings

DB = settings.db_path

schema_sql = """
CREATE TABLE IF NOT EXISTS matches_manual (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    league TEXT,
    season TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    home_xg REAL,
    away_xg REAL,
    notes TEXT
);
"""

samples = [
    ("英超", "2023-24", "曼城", "阿森纳", 2, 1),
    ("英超", "2023-24", "切尔西", "利物浦", 1, 1),
    ("英超", "2023-24", "曼联", "曼城", 0, 3),
    ("英超", "2023-24", "阿森纳", "切尔西", 3, 0),
    ("英超", "2023-24", "利物浦", "热刺", 2, 2),
    ("英超", "2023-24", "热刺", "曼联", 1, 0),
    ("英超", "2023-24", "曼城", "切尔西", 4, 2),
    ("英超", "2023-24", "阿森纳", "利物浦", 2, 2),
    ("英超", "2023-24", "曼联", "切尔西", 1, 1),
    ("英超", "2023-24", "热刺", "曼城", 0, 2),
]

# 生成日期序列（最近 N 天）
start_date = datetime.now() - timedelta(days=len(samples))
dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(samples))]

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute(schema_sql)

ins_sql = (
    "INSERT INTO matches_manual (date, league, season, home_team, away_team, home_goals, away_goals) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)

for i, (league, season, ht, at, hg, ag) in enumerate(samples):
    cur.execute(ins_sql, (dates[i], league, season, ht, at, hg, ag))

conn.commit()
print(f"[seed] 插入演示比赛 {len(samples)} 场到 {DB}")
conn.close()