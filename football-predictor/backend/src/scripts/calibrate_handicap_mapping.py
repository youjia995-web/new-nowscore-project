import json, sqlite3, os
from statistics import median
from typing import List, Dict, Tuple

DB_PATH = os.environ.get("DB_PATH", "football_analysis.db")

def connect():
    path = DB_PATH
    if not os.path.isabs(path):
        # 采用 Settings 默认的绝对路径更稳妥；此处保留相对兼容
        base_dir = os.path.dirname(__file__)
        # 退回到仓库根（scripts 位于 backend/src/scripts）
        root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".."))
        path = os.path.join(root, "football_analysis.db")
    return sqlite3.connect(path)

def expected_goal_diff(score_matrix: List[List[float]]) -> float:
    if not score_matrix:
        return 0.0
    h = len(score_matrix)
    w = len(score_matrix[0])
    egd = 0.0
    total = 0.0
    for i in range(h):
        for j in range(w):
            p = float(score_matrix[i][j])
            egd += (i - j) * p
            total += p
    if total > 0:
        egd /= total
    return float(egd)

def run():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT result_json FROM prediction_logs WHERE result_json IS NOT NULL")
    rows = cur.fetchall()
    pairs: List[Tuple[float, float]] = []  # (abs(egd), abs(market_final_handicap))
    for (js,) in rows:
        try:
            res = json.loads(js)
            sm = res.get("score_matrix")
            mfh = res.get("market_final_handicap")
            if sm and mfh is not None:
                egd = expected_goal_diff(sm)
                pairs.append((abs(egd), abs(float(mfh))))
        except Exception:
            continue
    if len(pairs) < 50:
        print("样本不足，至少需要 50 条预测日志。")
        return
    pairs.sort(key=lambda x: x[0])
    bins = 9
    bin_size = max(1, len(pairs) // bins)
    edges_lines: List[Dict[str, float]] = []
    for b in range(bins):
        lo = b * bin_size
        hi = min((b + 1) * bin_size, len(pairs))
        if lo >= hi:
            break
        xs = [pairs[k][0] for k in range(lo, hi)]
        ys = [pairs[k][1] for k in range(lo, hi)]
        edge = max(xs)
        med = median(ys)
        line_q = round(med * 4.0) / 4.0
        edges_lines.append({"edge": float(edge), "line": float(line_q)})
    cleaned: List[Dict[str, float]] = []
    for el in edges_lines:
        if not cleaned or abs(el["line"] - cleaned[-1]["line"]) >= 1e-6:
            cleaned.append(el)
        else:
            cleaned[-1]["edge"] = max(cleaned[-1]["edge"], el["edge"])
    if cleaned:
        cleaned[-1]["edge"] = max(cleaned[-1]["edge"], 9.99)
    else:
        cleaned = [{"edge": 9.99, "line": 2.00}]
    payload = {"piecewise": cleaned}
    cur.execute(
        """
        INSERT INTO calibration_params (key, value_json, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """,
        ("handicap_mapping_v1", json.dumps(payload, ensure_ascii=False))
    )
    conn.commit()
    conn.close()
    print("已写入 handicap_mapping_v1：", payload)

if __name__ == "__main__":
    run()