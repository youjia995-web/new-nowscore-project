import os, json, sqlite3, math
from typing import Optional, Dict, Any, List

class CalibrationService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        path = self.db_path
        if not os.path.isabs(path):
            base_dir = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(base_dir, path))
        conn = sqlite3.connect(path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        return conn

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT value_json FROM calibration_params WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO calibration_params (key, value_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value, ensure_ascii=False))
        )
        conn.commit()
        conn.close()

    def compute_market_weight(self, company_count: int, stability: float, model_entropy: float, score_peak: float) -> tuple[float, str]:
        """学习型融合权重：sigmoid线性函数映射到 [w_min,w_max]。"""
        params = self.get("fusion_weight_v1") or {
            "w_min": 0.30, "w_max": 0.70,
            "intercept": 0.00,
            "coef_entropy": 0.15,
            "coef_score_peak": -0.10,
            "coef_stability": 0.10,
            "coef_company_count": 0.04,
            "mode": "geometric"
        }
        w_min = float(params.get("w_min", 0.30))
        w_max = float(params.get("w_max", 0.70))
        b0 = float(params.get("intercept", 0.00))
        c_ent = float(params.get("coef_entropy", 0.15))
        c_peak = float(params.get("coef_score_peak", -0.10))
        c_stab = float(params.get("coef_stability", 0.10))
        c_cnt = float(params.get("coef_company_count", 0.04))
        reasons: List[str] = []

        max_ent = math.log(3)
        ent_norm = max(0.0, min(1.0, (model_entropy or 0.0) / max_ent))
        stab = max(0.0, min(1.0, stability or 0.0))
        cnt_norm = max(0.0, min(1.0, (company_count or 0) / 6.0))

        z = b0 + c_ent * ent_norm + c_peak * (score_peak or 0.0) + c_stab * stab + c_cnt * cnt_norm
        w = 1.0 / (1.0 + math.exp(-z))
        w = w_min + (w_max - w_min) * w

        reasons.append(f"entropy={ent_norm:.2f}")
        reasons.append(f"score_peak={score_peak:.2f}")
        reasons.append(f"stability={stab:.2f}")
        reasons.append(f"companies={company_count}")
        return float(w), "；".join(reasons) + f" → 学习权重={w:.2f}（范围{w_min:.2f}-{w_max:.2f}）"

    def map_goal_diff_to_handicap(self, goal_diff: float, league: Optional[str] = None) -> float:
        """净胜球→让球线映射：优先使用分段标定，否则回退旧阈值。"""
        cfg = self.get("handicap_mapping_v1")
        x = float(goal_diff or 0.0)
        sgn = 1.0 if x >= 0 else -1.0
        ax = abs(x)
        piece = None
        if cfg and "piecewise" in cfg:
            piece = cfg["piecewise"]
        else:
            piece = [
                {"edge": 0.10, "line": 0.00},
                {"edge": 0.30, "line": 0.25},
                {"edge": 0.50, "line": 0.50},
                {"edge": 0.80, "line": 0.75},
                {"edge": 1.10, "line": 1.00},
                {"edge": 1.40, "line": 1.25},
                {"edge": 1.70, "line": 1.50},
                {"edge": 2.00, "line": 1.75},
                {"edge": 9.99, "line": 2.00},
            ]
        for seg in piece:
            if ax < float(seg["edge"]):
                return sgn * float(seg["line"])
        return sgn * float(piece[-1]["line"])