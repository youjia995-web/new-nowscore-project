#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 team_aliases 表将数据库内的中文队名批量替换为英文规范名。

更新范围（主库）：
- team_metrics_daily.team
- team_season_stats.team
- prediction_logs.home_team / prediction_logs.away_team
- matches_manual.home_team / matches_manual.away_team（若存在）
- matches_data.home_team / matches_data.away_team（若存在）

更新范围（阵容库 rosters.db）：
- team_roster.team（若存在）

说明：
- 依赖 backend/src/config.py 的 settings 以获取实际数据库路径（考虑环境变量覆盖）。
- 使用 team_aliases 的映射（canonical_name=英文，alias_name=中文）。
- 仅当表与列存在时才执行更新；统计每张表的影响行数。
"""

import os
import sqlite3
from pathlib import Path
from importlib import import_module
from typing import Dict, List, Tuple


def _abs(path: str, base: Path) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (base / p).resolve())


def load_settings():
    # backend/src/scripts -> backend/src
    base = Path(__file__).resolve().parents[1]
    import sys
    sys.path.append(str(base))
    settings = import_module("config").settings
    return settings, base


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (table,))
    return cur.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols
    except Exception:
        return False


def load_alias_map(conn: sqlite3.Connection) -> Dict[str, str]:
    """返回 {alias_name(中文): canonical_name(英文)} 映射。"""
    cur = conn.cursor()
    if not table_exists(conn, "team_aliases"):
        return {}
    cur.execute("SELECT alias_name, canonical_name FROM team_aliases")
    mp: Dict[str, str] = {}
    for alias, canon in cur.fetchall():
        a = (str(alias or "").strip()).replace(" ", "")
        c = str(canon or "").strip()
        if a and c:
            mp[a] = c
    return mp


def update_single_column(conn: sqlite3.Connection, table: str, column: str, mp: Dict[str, str]) -> int:
    """逐条执行 UPDATE table SET column=? WHERE column=?，返回受影响行数。"""
    if not table_exists(conn, table) or not column_exists(conn, table, column):
        return 0
    cur = conn.cursor()
    total = 0
    for alias, canon in mp.items():
        # 使用 TRIM 并去掉空格的兼容匹配（尽量覆盖导入时未清洗的情况）
        # 先尝试精确匹配
        cur.execute(f"UPDATE {table} SET {column}=? WHERE {column}=?", (canon, alias))
        total += cur.rowcount or 0
        # 再尝试去空格匹配（仅当空格版不同）
        nospace_alias = alias.replace(" ", "")
        if nospace_alias != alias:
            cur.execute(f"UPDATE {table} SET {column}=? WHERE REPLACE({column}, ' ', '')=?", (canon, nospace_alias))
            total += cur.rowcount or 0
    conn.commit()
    return total


def migrate_main_db(db_path: str) -> List[Tuple[str, int]]:
    conn = connect(db_path)
    try:
        mp = load_alias_map(conn)
        if not mp:
            return [("team_aliases(empty)", 0)]
        stats: List[Tuple[str, int]] = []
        stats.append(("team_metrics_daily.team", update_single_column(conn, "team_metrics_daily", "team", mp)))
        stats.append(("team_season_stats.team", update_single_column(conn, "team_season_stats", "team", mp)))
        stats.append(("prediction_logs.home_team", update_single_column(conn, "prediction_logs", "home_team", mp)))
        stats.append(("prediction_logs.away_team", update_single_column(conn, "prediction_logs", "away_team", mp)))
        # 可选的比赛原始表
        stats.append(("matches_manual.home_team", update_single_column(conn, "matches_manual", "home_team", mp)))
        stats.append(("matches_manual.away_team", update_single_column(conn, "matches_manual", "away_team", mp)))
        stats.append(("matches_data.home_team", update_single_column(conn, "matches_data", "home_team", mp)))
        stats.append(("matches_data.away_team", update_single_column(conn, "matches_data", "away_team", mp)))
        return stats
    finally:
        conn.close()


def migrate_roster_db(db_path: str, alias_map: Dict[str, str]) -> List[Tuple[str, int]]:
    conn = connect(db_path)
    try:
        stats: List[Tuple[str, int]] = []
        if not alias_map:
            # 若没有映射表，保守跳过；避免误改
            return [("team_roster.team(skipped_no_alias_map)", 0)]
        stats.append(("team_roster.team", update_single_column(conn, "team_roster", "team", alias_map)))
        return stats
    finally:
        conn.close()


def main():
    settings, base = load_settings()
    main_db = _abs(settings.db_path, base)
    roster_db = _abs(settings.roster_db_path, base)
    print(f"[paths] main_db={main_db}")
    print(f"[paths] roster_db={roster_db}")
    print("[migrate] start main db")
    main_stats = migrate_main_db(main_db)
    for name, cnt in main_stats:
        print(f"[updated] {name}: {cnt}")
    # 从主库重新加载 alias 映射用于阵容库
    alias_map: Dict[str, str] = {}
    try:
        alias_map = load_alias_map(connect(main_db))
    except Exception:
        alias_map = {}
    print("[migrate] start roster db")
    roster_stats = migrate_roster_db(roster_db, alias_map)
    for name, cnt in roster_stats:
        print(f"[updated] {name}: {cnt}")
    print("[done] migration completed")


if __name__ == "__main__":
    main()