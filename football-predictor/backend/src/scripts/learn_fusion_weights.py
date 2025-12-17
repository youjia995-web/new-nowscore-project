import json, sqlite3, os, math
from typing import List, Dict

DB_PATH = os.environ.get("DB_PATH", "football_analysis.db")

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def fuse_probs(model: Dict[str, float], market: Dict[str, float], w: float, mode: str = "geometric") -> Dict[str, float]:
    eps = 1e-12
    if mode == "geometric":
        hm = math.exp((1 - w) * math.log(max(model["home_win"], eps)) + w * math.log(max(market["home_win"], eps)))
        dr = math.exp((1 - w) * math.log(max(model["draw"], eps)) + w * math.log(max(market["draw"], eps)))
        aw = math.exp((1 - w) * math.log(max(model["away_win"], eps)) + w * math.log(max(market["away_win"], eps)))
        s = hm + dr + aw
        return {"home_win": hm / s, "draw": dr / s, "away_win": aw / s}
    else:
        return {
            "home_win": (1 - w) * model["home_win"] + w * market["home_win"],
            "draw": (1 - w) * model["draw"] + w * market["draw"],
            "away_win": (1 - w) * model["away_win"] + w * market["away_win"],
        }

def connect():
    path = DB_PATH
    if not os.path.isabs(path):
        base_dir = os.path.dirname(__file__)
        root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".."))
        path = os.path.join(root, "football_analysis.db")
    return sqlite3.connect(path)

def load_samples(conn) -> List[Dict]:
    cur = conn.cursor()
    cur.execute("SELECT result_json, actual_outcome FROM prediction_logs WHERE actual_outcome IS NOT NULL")
    rows = cur.fetchall()
    samples: List[Dict] = []
    for js, ao in rows:
        try:
            res = json.loads(js)
            model = {
                "home_win": float(res["home_win_prob"]),
                "draw": float(res["draw_prob"]),
                "away_win": float(res["away_win_prob"]),
            }
            market = {
                "home_win": float(res["market_home_win_prob"]),
                "draw": float(res["market_draw_prob"]),
                "away_win": float(res["market_away_win_prob"]),
            }
            cm = res.get("calibration_metrics") or {}
            ent = float(cm.get("outcome_entropy", 0.0))
            peak = float(cm.get("score_peak_prob", 0.0))
            samples.append({"model": model, "market": market, "entropy": ent, "peak": peak, "actual": ao})
        except Exception:
            continue
    return samples

def optimize(samples: List[Dict], steps: int = 600, lr: float = 0.05, w_min: float = 0.30, w_max: float = 0.70) -> Dict:
    # w = w_min + (w_max - w_min) * sigmoid(b0 + b1*ent_norm + b2*peak)
    b0, b1, b2 = 0.0, 0.15, -0.10
    mode = "geometric"
    max_ent = math.log(3)
    for t in range(steps):
        g0 = g1 = g2 = 0.0
        loss = 0.0
        for s in samples:
            ent_norm = max(0.0, min(1.0, s["entropy"] / max_ent))
            peak = s["peak"]
            z = b0 + b1 * ent_norm + b2 * peak
            sig = sigmoid(z)
            w = w_min + (w_max - w_min) * sig
            fused = fuse_probs(s["model"], s["market"], w, mode=mode)
            actual = {"H": "home_win", "D": "draw", "A": "away_win"}[s["actual"]]
            p = fused[actual]
            p = max(p, 1e-9)
            loss += -math.log(p)
            # 近似线性融合梯度
            diff = s["market"][actual] - s["model"][actual]
            dL_dw = -(1.0 / p) * diff
            dw_dsig = (w_max - w_min)
            dsig_dz = sig * (1.0 - sig)
            dL_dz = dL_dw * dw_dsig * dsig_dz
            g0 += dL_dz
            g1 += dL_dz * ent_norm
            g2 += dL_dz * peak
        b0 -= lr * g0 / max(1, len(samples))
        b1 -= lr * g1 / max(1, len(samples))
        b2 -= lr * g2 / max(1, len(samples))
        if t % 100 == 0:
            print(f"step={t}, avg_logloss={loss/len(samples):.4f}, b0={b0:.3f}, b1={b1:.3f}, b2={b2:.3f}")
    return {
        "w_min": w_min, "w_max": w_max,
        "intercept": b0,
        "coef_entropy": b1,
        "coef_score_peak": b2,
        "coef_stability": 0.0,
        "coef_company_count": 0.0,
        "mode": mode
    }

def run():
    conn = connect()
    samples = load_samples(conn)
    if len(samples) < 200:
        print("样本不足，至少需要 200 条带赛果的预测日志。")
        conn.close()
        return
    params = optimize(samples)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO calibration_params (key, value_json, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
        """,
        ("fusion_weight_v1", json.dumps(params, ensure_ascii=False))
    )
    conn.commit()
    conn.close()
    print("已写入 fusion_weight_v1：", params)

if __name__ == "__main__":
    run()