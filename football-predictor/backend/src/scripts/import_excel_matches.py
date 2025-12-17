#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从指定 Excel 导入比赛到 SQLite：优先写入原库 `matches_data`，否则回退到 `matches_manual`。
支持常见中文/英文列名自动识别：日期/联赛/赛季/轮次/主队/客队/比分/半场比分/主客进球/主客xG/备注。
示例：
  python3 import_excel_matches.py \
      /Users/huojin/Desktop/通用型足彩分析/英超2024-2025.xlsx \
      --league 英超 --season 2024-2025 --sheet 0
"""
import sys, os, re
from pathlib import Path
from importlib import import_module
from typing import Dict, Optional

import pandas as pd
import sqlite3

# 定位 backend/src 并加载配置
BASE = Path(__file__).resolve().parents[1]
# 将 backend 根目录加入 sys.path，以便使用包名 src.* 进行导入
sys.path.append(str(BASE.parents[0]))
settings = import_module("src.config").settings
from src.services.prediction_store import PredictionStore
from src.models.match import AdminMatch, TeamSeasonStats

DB = settings.db_path

# —— 列名识别 ——
KEYS = {
    "date": ["日期", "date", "比赛日期"],
    "league": ["联赛", "league", "unnamed_2"],
    "season": ["赛季", "season", "比赛信息"],
    "round": ["轮次", "round", "unnamed_3", "周"],
    "home_team": ["主队", "home", "主队名称", "基本面_2", "home_team"],
    "away_team": ["客队", "away", "客队名称", "基本面_9", "away_team"],
    "home_goals": ["主队进球", "fthg", "全场主队进球", "半全场数据_6", "hg"],
    "away_goals": ["客队进球", "ftag", "全场客队进球", "半全场数据_7", "ag"],
    "score": ["比分", "全场比分", "ft"],
    "ht_score": ["半场比分", "ht", "半全场数据_5"],
    "half_home_goals": ["半场主队进球", "hthg", "半全场数据_2"],
    "half_away_goals": ["半场客队进球", "htag", "半全场数据_3"],
    "home_xg": ["主队xg", "主xg", "主队预期进球", "xgh", "home_xg"],
    "away_xg": ["客队xg", "客xg", "客队预期进球", "xga_match", "away_xg"],
    "notes": ["备注", "notes"],
    # 球队赛季统计（用于 team_season_stats）
    "team_name": ["球队名称", "球队", "team", "name"],
    "xg_team": ["xg", "预期进球"],
    "xga_team": ["xga", "预期失球"],
    "xpts_team": ["xpts", "预期积分"],
}

def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "").strip().lower())

def find_col(df_cols, keys) -> Optional[str]:
    cols_norm = {norm(c): c for c in df_cols}
    for k in keys:
        k_norm = norm(k)
        # 完全相等或包含匹配
        if k_norm in cols_norm:
            return cols_norm[k_norm]
        for cn, orig in cols_norm.items():
            if k_norm and k_norm in cn:
                return orig
    return None

def parse_filename_defaults(path: Path) -> Dict[str, Optional[str]]:
    name = path.stem
    # 例：英超2024-2025 / 英超_2024-25
    m = re.search(r"^(?P<league>[^0-9]+)?(?P<season>\d{4}[\-_/]\d{2,4})", name)
    league = None
    season = None
    if m:
        league = (m.group("league") or "").strip()
        season = (m.group("season") or "").replace("_", "-").strip()
    return {"league": league or None, "season": season or None}

def to_iso_date(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        s = str(x).strip()
        m = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return None

def parse_score(s) -> (Optional[int], Optional[int]):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None, None
    text = str(s).strip()
    m = re.search(r"^(\d+)\s*[-:：]\s*(\d+)$", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def main():
    import argparse
    ap = argparse.ArgumentParser(description="导入 Excel 比赛数据到 SQLite")
    ap.add_argument("excel_path", help="Excel 文件绝对路径 (.xlsx/.xls)")
    ap.add_argument("--sheet", default=0, help="工作表名或索引（默认0）")
    ap.add_argument("--league", default=None, help="默认联赛（若表中无列）")
    ap.add_argument("--season", default=None, help="默认赛季（若表中无列）")
    args = ap.parse_args()

    fpath = Path(args.excel_path).expanduser()
    if not fpath.exists():
        print(f"[error] 文件不存在: {fpath}")
        sys.exit(2)

    # 读取 Excel
    try:
        df = pd.read_excel(fpath, sheet_name=args.sheet, engine="openpyxl")
    except Exception as e:
        # 无 openpyxl 时尝试默认引擎
        try:
            df = pd.read_excel(fpath, sheet_name=args.sheet)
        except Exception as e2:
            print(f"[error] 读取 Excel 失败，请安装 openpyxl 或检查文件格式: {e2}")
            sys.exit(3)

    # 推断联赛/赛季默认值
    defaults = parse_filename_defaults(fpath)
    default_league = args.league or defaults.get("league")
    default_season = args.season or defaults.get("season")

    # 列映射
    cols = df.columns.tolist()
    col_map = {k: find_col(cols, v) for k, v in KEYS.items()}

    # 打开存储
    store = PredictionStore(DB)
    # 建立重复检查连接
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    def table_exists(name: str) -> bool:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (name,))
        return cur.fetchone() is not None
    has_md = table_exists("matches_data")

    # 若识别到球队赛季列，则走球队赛季统计导入
    if col_map.get("team_name") and col_map.get("season") and col_map.get("xg_team") and col_map.get("xga_team") and col_map.get("xpts_team"):
        total = 0
        inserted = 0
        skipped = 0
        for _, row in df.iterrows():
            total += 1
            team = row[col_map["team_name"]]
            season = row[col_map["season"]]
            try:
                xg = float(row[col_map["xg_team"]])
                xga = float(row[col_map["xga_team"]])
                xpts = float(row[col_map["xpts_team"]])
            except Exception:
                skipped += 1
                continue
            if not team or not season:
                skipped += 1
                continue
            try:
                store.upsert_team_season_stats(TeamSeasonStats(
                    team=str(team).strip(),
                    season=str(season).strip(),
                    xg=float(xg),
                    xga=float(xga),
                    xpts=float(xpts),
                ))
                inserted += 1
            except Exception as e:
                skipped += 1
                print(f"[warn] 赛季统计写入跳过: {e}")
        print(f"[done] 球队赛季统计导入完成: {fpath}")
        print(f"[stats] 总行: {total}, 成功: {inserted}, 跳过: {skipped}")
        print(f"[db] 路径: {DB}")
        try:
            conn.close()
        except Exception:
            pass
        return

    total = 0
    inserted = 0
    skipped = 0
    for _, row in df.iterrows():
        total += 1
        def get(col_key):
            c = col_map.get(col_key)
            return (row[c] if c in row else None)
        date = to_iso_date(get("date"))
        league = (get("league") or default_league)
        season = (get("season") or default_season)
        round_ = get("round")
        ht = get("home_team")
        at = get("away_team")
        hg = get("home_goals")
        ag = get("away_goals")
        # 先做重复检查
        try:
            if ht and at:
                ht_s = str(ht).strip()
                at_s = str(at).strip()
                if has_md:
                    if season:
                        cur.execute("SELECT 1 FROM matches_data WHERE 基本面_2=? AND 基本面_9=? AND 比赛信息=? LIMIT 1", (ht_s, at_s, str(season).strip()))
                    else:
                        cur.execute("SELECT 1 FROM matches_data WHERE 基本面_2=? AND 基本面_9=? LIMIT 1", (ht_s, at_s))
                    if cur.fetchone():
                        skipped += 1
                        continue
                else:
                    if date:
                        cur.execute("SELECT 1 FROM matches_manual WHERE home_team=? AND away_team=? AND date=? LIMIT 1", (ht_s, at_s, date))
                    elif season:
                        cur.execute("SELECT 1 FROM matches_manual WHERE home_team=? AND away_team=? AND season=? LIMIT 1", (ht_s, at_s, str(season).strip()))
                    else:
                        cur.execute("SELECT 1 FROM matches_manual WHERE home_team=? AND away_team=? LIMIT 1", (ht_s, at_s))
                    if cur.fetchone():
                        skipped += 1
                        continue
        except Exception:
            # 重复检查失败则忽略，继续插入尝试
            pass
        if (hg is None or (isinstance(hg, float) and pd.isna(hg))) and col_map.get("score"):
            h2, a2 = parse_score(get("score"))
            hg, ag = (hg if hg is not None else h2), (ag if ag is not None else a2)
        hhg = get("half_home_goals")
        hag = get("half_away_goals")
        if (hhg is None or (isinstance(hhg, float) and pd.isna(hhg))) and col_map.get("ht_score"):
            hht, aht = parse_score(get("ht_score"))
            hhg, hag = (hhg if hhg is not None else hht), (hag if hag is not None else aht)
        hxg = get("home_xg")
        axg = get("away_xg")
        notes = get("notes")

        # 必填项校验
        if not ht or not at:
            skipped += 1
            continue
        # 整型/浮点转换
        def to_int(v):
            try:
                return int(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None
            except Exception:
                return None
        def to_float(v):
            try:
                return float(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None
            except Exception:
                return None
        hg, ag = to_int(hg), to_int(ag)
        hhg, hag = to_int(hhg), to_int(hag)
        hxg, axg = to_float(hxg), to_float(axg)
        round_s = str(round_).strip() if round_ is not None and str(round_).strip() != "" else None
        notes_s = str(notes).strip() if notes is not None and str(notes).strip() != "" else None

        match = AdminMatch(
            date=date,
            league=(str(league).strip() if league else None),
            season=(str(season).strip() if season else None),
            home_team=str(ht).strip(),
            away_team=str(at).strip(),
            home_goals=hg,
            away_goals=ag,
            home_xg=hxg,
            away_xg=axg,
            round=round_s,
            half_home_goals=hhg,
            half_away_goals=hag,
            notes=notes_s,
        )
        try:
            store.insert_match_original(match)
            inserted += 1
        except Exception as e:
            skipped += 1
            print(f"[warn] 插入跳过: {e}")

    print(f"[done] 文件: {fpath}")
    print(f"[stats] 总行: {total}, 成功: {inserted}, 跳过: {skipped}")
    print(f"[db] 路径: {DB}")
    try:
        conn.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()