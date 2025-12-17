import argparse
import sqlite3
from typing import Dict, List, Tuple

import pandas as pd


def _detect_columns(cols: List[str]) -> Tuple[str, str]:
    cn_candidates = {"中文队名", "中文名称", "name_cn", "cn_name", "中文", "队名(中文)", "队名_中文"}
    en_candidates = {"英文队名", "英文名称", "name_en", "en_name", "英文", "队名(英文)", "队名_英文", "队名英文"}

    cols_lower = {c.lower(): c for c in cols}
    cn = None
    en = None

    for c in cols:
        if c in cn_candidates:
            cn = c
            break
    for c in cols:
        if c in en_candidates:
            en = c
            break

    if cn and en:
        return cn, en

    fallbacks_cn = [
        "中文队名",
        "中文",
        "队名",
        "name_cn",
        "cn_name",
        "球队中文",
    ]
    fallbacks_en = [
        "英文队名",
        "英文",
        "英文名",
        "队名英文",
        "name_en",
        "en_name",
        "英语",
    ]

    for key in fallbacks_cn:
        if key in cols:
            cn = key
            break
    for key in fallbacks_en:
        if key in cols:
            en = key
            break

    if not cn or not en:
        raise ValueError("无法识别Excel中的中文/英文列名，请在表头使用标准列名")
    return cn, en


def load_mapping(xlsx_path: str, sheet_name: str | None = None) -> Dict[str, str]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    if isinstance(df, dict):
        df = next(iter(df.values()))
    cn_col, en_col = _detect_columns(list(df.columns))
    mapping = {}
    for _, row in df.iterrows():
        cn = str(row[cn_col]).strip() if pd.notna(row[cn_col]) else ""
        en = str(row[en_col]).strip() if pd.notna(row[en_col]) else ""
        if cn and en:
            mapping[cn] = en
    if not mapping:
        raise ValueError("转换表为空，未读取到任何有效的中英文映射")
    return mapping


def _get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def update_teams_table(conn: sqlite3.Connection, mapping: Dict[str, str], dry_run: bool = False) -> Tuple[int, List[str]]:
    updated = 0
    missing: List[str] = []

    try:
        cols = _get_columns(conn, "teams")
    except sqlite3.Error:
        return 0, []

    target_cols: List[str] = []
    if "name_en" in cols:
        target_cols.append("name_en")
    if "name" in cols:
        target_cols.append("name")

    if not target_cols:
        return 0, []

    cur = conn.execute("SELECT rowid, name FROM teams") if "name" in cols else conn.execute("SELECT rowid, name_en FROM teams")
    rows = cur.fetchall()
    for rowid, current in rows:
        if current is None:
            continue
        cn = str(current).strip()
        en = mapping.get(cn)
        if en:
            for col in target_cols:
                if dry_run:
                    continue
                conn.execute(f"UPDATE teams SET {col} = ? WHERE rowid = ?", (en, rowid))
            updated += 1
        else:
            missing.append(cn)

    return updated, missing


def update_matches_table(conn: sqlite3.Connection, mapping: Dict[str, str], dry_run: bool = False) -> Tuple[int, List[str]]:
    updated = 0
    missing: List[str] = []

    try:
        cols = _get_columns(conn, "matches_data")
    except sqlite3.Error:
        return 0, []

    # 支持中英文主客队列名，优先匹配更规范的列名
    def _pick_col(available: List[str], candidates: List[str]) -> str | None:
        for c in candidates:
            if c in available:
                return c
        return None

    home_col = _pick_col(
        cols,
        [
            "home_team",
            "HomeTeam",
            "home",
            "home_name",
            "team_home",
            "homeTeam",
            "主队",
            "主队名称",
            # matches_data 观测到的主队名称列
            "基本面_2",
        ],
    )
    away_col = _pick_col(
        cols,
        [
            "away_team",
            "AwayTeam",
            "away",
            "away_name",
            "team_away",
            "awayTeam",
            "客队",
            "客队名称",
            # matches_data 观测到的客队名称列
            "基本面_9",
        ],
    )
    if not home_col or not away_col:
        return 0, []

    cur = conn.execute(f"SELECT rowid, {home_col}, {away_col} FROM matches_data")
    rows = cur.fetchall()
    for rowid, home, away in rows:
        new_home = mapping.get(str(home).strip()) if home is not None else None
        new_away = mapping.get(str(away).strip()) if away is not None else None
        if new_home or new_away:
            if not dry_run:
                conn.execute(
                    f"UPDATE matches_data SET {home_col} = COALESCE(?, {home_col}), {away_col} = COALESCE(?, {away_col}) WHERE rowid = ?",
                    (new_home, new_away, rowid),
                )
            updated += 1
        if home and str(home).strip() not in mapping:
            missing.append(str(home).strip())
        if away and str(away).strip() not in mapping:
            missing.append(str(away).strip())

    return updated, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="SQLite 数据库文件路径")
    parser.add_argument("--xlsx", required=True, help="队名中英文对照表.xlsx 路径")
    parser.add_argument("--sheet", default=None, help="工作表名称，可选")
    parser.add_argument("--dry-run", action="store_true", help="仅预览不写入")
    args = parser.parse_args()

    mapping = load_mapping(args.xlsx, args.sheet)
    conn = sqlite3.connect(args.db)
    try:
        conn.execute("BEGIN")
        t_updated, t_missing = update_teams_table(conn, mapping, dry_run=args.dry_run)
        m_updated, m_missing = update_matches_table(conn, mapping, dry_run=args.dry_run)
        if args.dry_run:
            conn.execute("ROLLBACK")
        else:
            conn.execute("COMMIT")
    finally:
        conn.close()

    missing = sorted(set((t_missing or []) + (m_missing or [])))
    print("Teams updated:", t_updated)
    print("Matches updated:", m_updated)
    print("Unmatched team names:", len(missing))
    for name in missing[:100]:
        print(" -", name)


if __name__ == "__main__":
    main()