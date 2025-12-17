#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小联赛参数标定脚本
- 读取 SQLite 中的 `matches_manual` 和 `team_season_stats`
- 使用 PoissonEngine 进行纯模型概率计算（不依赖市场赔率）
- 网格搜索 `league_base_rate`、`home_advantage`、`league_tempo`、`dc_rho`
- 目标函数：平均对数损失（logloss），同时输出 Brier 与准确率

运行示例：
    python3 backend/src/scripts/calibrate_league_params.py --season 2023-24 --league 英超

说明：
- 若某球队缺少该赛季的 xG/xGA/xPTS，则回退到该队历史均值；再缺失则使用通用缺省值。
- 近况（最近6场）的进/失球缺失时，用赛季场均 * 6 的估计值。
- games_played 无显式字段，默认使用 38（五大联赛常规场次）。
"""

import os
import sys
import math
import sqlite3
import argparse
from typing import Dict, Tuple, List, Optional

# 将 backend 目录加入路径，使用隐式命名空间包导入 src.*
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
sys.path.append(BACKEND)

# 将 backend/src 加入路径，直接使用顶层模块导入 models 与 config
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(SRC_DIR)

# 动态载入 PoissonEngine，修正相对导入为绝对导入
import importlib.util
import types

def _load_poisson_engine_class():
    path = os.path.join(SRC_DIR, 'engines', 'poisson_engine.py')
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    code = code.replace('from ..models.match', 'from models.match').replace('from ..config', 'from config')
    module = types.ModuleType('poisson_engine_hacked')
    exec(compile(code, path, 'exec'), module.__dict__)
    return module.PoissonEngine

from models.match import TeamStats
from config import settings

PoissonEngine = _load_poisson_engine_class()

Outcome = str

def _actual_outcome(home_goals: int, away_goals: int) -> Outcome:
    if home_goals > away_goals:
        return 'H'
    if home_goals < away_goals:
        return 'A'
    return 'D'

def _fetch_team_season_stats(conn: sqlite3.Connection, team: str, season: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    cur = conn.cursor()
    if season:
        cur.execute("SELECT xg, xga, xpts FROM team_season_stats WHERE team=? AND season=?", (team, season))
        row = cur.fetchone()
        if row:
            try:
                return float(row[0]), float(row[1]), float(row[2])
            except Exception:
                pass
    # 回退：该队历史均值
    cur.execute("SELECT AVG(xg), AVG(xga), AVG(xpts) FROM team_season_stats WHERE team=?", (team,))
    row = cur.fetchone()
    if row and row[0] is not None and row[1] is not None and row[2] is not None:
        return float(row[0]), float(row[1]), float(row[2])
    return None, None, None

def _estimate_ranking(conn: sqlite3.Connection, team: str, season: Optional[str]) -> int:
    try:
        if not season:
            return 10
        cur = conn.cursor()
        cur.execute("SELECT team, xpts FROM team_season_stats WHERE season=?", (season,))
        rows = cur.fetchall()
        if not rows:
            return 10
        sorted_rows = sorted(rows, key=lambda r: (r[1] if r[1] is not None else -1e9), reverse=True)
        for idx, (t, _) in enumerate(sorted_rows, start=1):
            if str(t).strip() == str(team).strip():
                return idx
        return 10
    except Exception:
        return 10

def _build_team_stats(conn: sqlite3.Connection, team: str, season: Optional[str]) -> TeamStats:
    xg, xga, xpts = _fetch_team_season_stats(conn, team, season)
    # 缺省值：保守中性（赛季总值）
    if xg is None or xga is None:
        # 近似：五大联赛平均每队 xG/xGA ~ 50 左右
        xg = 50.0 if xg is None else xg
        xga = 50.0 if xga is None else xga
    if xpts is None:
        xpts = 50.0
    gp = 38  # 缺省场次
    # 近况估计：场均 * 6
    recent_scored = int(round((xg / gp) * 6))
    recent_conceded = int(round((xga / gp) * 6))
    ranking = _estimate_ranking(conn, team, season)
    return TeamStats(
        name=team,
        ranking=ranking,
        games_played=gp,
        xg=float(xg),
        xga=float(xga),
        xpts=float(xpts),
        recent_goals_scored=recent_scored,
        recent_goals_conceded=recent_conceded,
        npxg=None,
        npxga=None,
        ppda=None,
        oppda=None,
        dc=None,
        odc=None,
    )

def _load_matches(conn: sqlite3.Connection, season: Optional[str], league: Optional[str], limit: Optional[int]) -> List[Dict]:
    cur = conn.cursor()
    sql = "SELECT date, league, season, home_team, away_team, home_goals, away_goals FROM matches_manual WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL"
    params: List = []
    if season:
        sql += " AND season = ?"
        params.append(season)
    if league:
        sql += " AND league = ?"
        params.append(league)
    sql += " ORDER BY date DESC"
    if isinstance(limit, int) and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    items = []
    for date, lg, ssn, ht, at, hg, ag in rows:
        items.append({
            "date": date,
            "league": lg,
            "season": ssn,
            "home_team": ht,
            "away_team": at,
            "home_goals": int(hg),
            "away_goals": int(ag),
            "actual": _actual_outcome(int(hg), int(ag)),
        })
    return items

def _metrics_for_outcome(dist: Dict[str, float], actual: Outcome) -> Tuple[float, float, bool]:
    ph, pd, pa = dist.get("home_win", 0.0), dist.get("draw", 0.0), dist.get("away_win", 0.0)
    # Brier（三类）
    oh, od, oa = (1.0, 0.0, 0.0) if actual == 'H' else ((0.0, 1.0, 0.0) if actual == 'D' else (0.0, 0.0, 1.0))
    brier = (ph - oh) ** 2 + (pd - od) ** 2 + (pa - oa) ** 2
    # Logloss
    p_actual = { 'H': ph, 'D': pd, 'A': pa }[actual]
    logloss = -math.log(max(p_actual, 1e-9))
    # 命中（峰值类别是否等于实际）
    pred_label = 'H' if ph >= pd and ph >= pa else ('D' if pd >= ph and pd >= pa else 'A')
    hit = (pred_label == actual)
    return brier, logloss, hit

def grid_search(conn: sqlite3.Connection, matches: List[Dict], grid: Dict[str, List[float]]) -> Dict:
    best = None
    engine = PoissonEngine()
    total = len(matches)
    for base in grid.get("league_base_rate", [1.0]):
        settings.league_base_rate = float(base)
        for ha in grid.get("home_advantage", [1.05]):
            settings.home_advantage = float(ha)
            for tempo in grid.get("league_tempo", [1.0]):
                settings.league_tempo = float(tempo)
                for rho in grid.get("dc_rho", [-0.12]):
                    settings.dc_rho = float(rho)
                    # 逐场评估
                    b_sum = 0.0
                    ll_sum = 0.0
                    hits = 0
                    for m in matches:
                        ht = _build_team_stats(conn, m["home_team"], m["season"]) 
                        at = _build_team_stats(conn, m["away_team"], m["season"]) 
                        dist, _, _ = engine.calculate_match_probabilities(ht, at)
                        b, ll, hit = _metrics_for_outcome(dist, m["actual"]) 
                        b_sum += b
                        ll_sum += ll
                        hits += (1 if hit else 0)
                    avg_b = b_sum / max(total, 1)
                    avg_ll = ll_sum / max(total, 1)
                    acc = hits / max(total, 1)
                    rec = {
                        "params": {"league_base_rate": base, "home_advantage": ha, "league_tempo": tempo, "dc_rho": rho},
                        "avg_brier": avg_b,
                        "avg_logloss": avg_ll,
                        "accuracy": acc,
                        "samples": total,
                    }
                    if (best is None) or (rec["avg_logloss"] < best["avg_logloss"]):
                        best = rec
    return best or {"error": "no_matches"}

def main():
    parser = argparse.ArgumentParser(description="联赛参数标定（最小版）")
    parser.add_argument("--season", type=str, default=None, help="赛季过滤，如 2023-24")
    parser.add_argument("--league", type=str, default=None, help="联赛过滤，如 英超")
    parser.add_argument("--limit", type=int, default=None, help="最大样本数")
    parser.add_argument("--grid", type=str, default="small", choices=["small", "full"], help="网格规模：small/full")
    args = parser.parse_args()

    db_path = settings.db_path
    # 替换连接为使用绝对 db_path
    conn = sqlite3.connect(settings.db_path)
    
    # 新增：从 prediction_logs 回退加载比赛数据
    
    def _load_matches_from_logs(conn: sqlite3.Connection, limit: Optional[int]) -> List[Dict]:
        cur = conn.cursor()
        sql = "SELECT created_at, home_team, away_team, actual_home_goals, actual_away_goals FROM prediction_logs WHERE actual_outcome IS NOT NULL ORDER BY created_at DESC"
        params: List = []
        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        items = []
        for created_at, ht, at, hg, ag in rows:
            if hg is None or ag is None:
                continue
            hg_i, ag_i = int(hg), int(ag)
            items.append({
                "date": created_at,
                "league": None,
                "season": None,
                "home_team": ht,
                "away_team": at,
                "home_goals": hg_i,
                "away_goals": ag_i,
                "actual": _actual_outcome(hg_i, ag_i),
            })
        return items
    
    try:
        matches = _load_matches(conn, args.season, args.league, args.limit)
        if not matches:
            matches = _load_matches_from_logs(conn, args.limit)
            if not matches:
                print("[calibrate] 无可用样本（matches_manual 与 prediction_logs 均为空）。")
                return
            else:
                print(f"[calibrate] 使用 prediction_logs 回退样本，共 {len(matches)} 场。")

        grid_small = {
            "league_base_rate": [0.96, 1.00, 1.04],
            "home_advantage": [1.02, 1.05, 1.08],
            "league_tempo": [0.96, 1.00, 1.04],
            "dc_rho": [-0.20, -0.15, -0.12, -0.10, -0.08],
        }
        grid_full = {
            "league_base_rate": [0.90, 0.96, 1.00, 1.04, 1.10],
            "home_advantage": [1.00, 1.03, 1.05, 1.07, 1.10],
            "league_tempo": [0.94, 0.98, 1.00, 1.02, 1.06],
            "dc_rho": [-0.25, -0.20, -0.15, -0.12, -0.10, -0.08],
        }
        grid = grid_small if args.grid == "small" else grid_full
        best = grid_search(conn, matches, grid)
        if best.get("error"):
            print("[calibrate] 未找到最佳参数：", best)
            return
        print("[calibrate] 最佳参数：", best["params"]) 
        print("[calibrate] 样本数：", best["samples"]) 
        print(f"[calibrate] 平均logloss：{best['avg_logloss']:.4f}，平均Brier：{best['avg_brier']:.4f}，准确率：{best['accuracy']:.3f}")
        # 提示如何应用
        print("[calibrate] 可通过环境变量覆盖：LEAGUE_BASE_RATE、HOME_ADVANTAGE、LEAGUE_TEMPO、DC_RHO")
    finally:
        conn.close()

if __name__ == "__main__":
    main()