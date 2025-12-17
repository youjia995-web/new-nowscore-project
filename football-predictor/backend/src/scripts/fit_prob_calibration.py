"""
基于历史预测与真实结果训练后验概率校准（Isotonic Regression），并写入 calibration_params。

运行示例：
  python3 backend/src/scripts/fit_prob_calibration.py --bins 51

要求：prediction_logs 表有实际赛果（actual_outcome in 'H','D','A'）。
写入键：prob_calibration_v1
"""

import os
import sqlite3
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _resolve_db_path() -> str:
    path = os.environ.get("DB_PATH", "football_analysis.db")
    if not os.path.isabs(path):
        # 退回到仓库根（scripts 位于 backend/src/scripts）
        base_dir = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".."))
        path = os.path.join(root, "football_analysis.db")
    return path


def _ensure_calibration_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value_json TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def load_training_data(conn: sqlite3.Connection) -> Dict[str, Tuple[List[float], List[int]]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pred_home, pred_draw, pred_away, actual_outcome
        FROM prediction_logs
        WHERE actual_outcome IS NOT NULL
          AND pred_home IS NOT NULL AND pred_draw IS NOT NULL AND pred_away IS NOT NULL
        """
    )
    rows = cur.fetchall()
    X_home, Y_home = [], []
    X_draw, Y_draw = [], []
    X_away, Y_away = [], []
    for ph, pd, pa, ao in rows:
        try:
            ph = float(ph); pd = float(pd); pa = float(pa)
        except Exception:
            continue
        if not (0.0 <= ph <= 1.0 and 0.0 <= pd <= 1.0 and 0.0 <= pa <= 1.0):
            continue
        if ao not in ("H", "D", "A"):
            continue
        X_home.append(ph); Y_home.append(1 if ao == "H" else 0)
        X_draw.append(pd); Y_draw.append(1 if ao == "D" else 0)
        X_away.append(pa); Y_away.append(1 if ao == "A" else 0)
    return {
        "home": (X_home, Y_home),
        "draw": (X_draw, Y_draw),
        "away": (X_away, Y_away),
    }


def fit_isotonic(points_in: List[float], labels: List[int], bins: int) -> List[List[float]]:
    n = len(points_in)
    # 若样本过少或标签退化，返回单位映射
    if n < 50 or len(set(labels)) < 2:
        xs = np.linspace(0.0, 1.0, bins)
        return [[float(x), float(x)] for x in xs]
    try:
        ir = IsotonicRegression(out_of_bounds="clip")
        xs = np.asarray(points_in, dtype=float)
        ys = np.asarray(labels, dtype=float)
        ir.fit(xs, ys)
        grid = np.linspace(0.0, 1.0, bins)
        pred = ir.predict(grid)
        # 上下界裁剪并与网格打包
        pred = np.clip(pred, 0.0, 1.0)
        return [[float(x), float(y)] for x, y in zip(grid, pred)]
    except Exception:
        xs = np.linspace(0.0, 1.0, bins)
        return [[float(x), float(x)] for x in xs]


def run(bins: int = 51):
    db_path = _resolve_db_path()
    conn = sqlite3.connect(db_path)
    _ensure_calibration_table(conn)
    data = load_training_data(conn)
    pts_home = fit_isotonic(*data["home"], bins)
    pts_draw = fit_isotonic(*data["draw"], bins)
    pts_away = fit_isotonic(*data["away"], bins)
    payload = {
        "points": {"home": pts_home, "draw": pts_draw, "away": pts_away},
        "meta": {"bin_count": int(bins), "source": "prediction_logs", "samples": {
            "home": len(data["home"][0]),
            "draw": len(data["draw"][0]),
            "away": len(data["away"][0]),
        }}
    }
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO calibration_params (key, value_json, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """,
        ("prob_calibration_v1", json.dumps(payload, ensure_ascii=False))
    )
    conn.commit()
    conn.close()
    print("已写入校准参数：prob_calibration_v1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=51, help="插值网格点数量（建议 51）")
    args = ap.parse_args()
    run(bins=args.bins)