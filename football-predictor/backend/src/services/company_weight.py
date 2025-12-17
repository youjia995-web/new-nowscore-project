from __future__ import annotations
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime

# 简单公司基础权重表（可随时扩充/覆盖）
BASE_WEIGHTS: Dict[str, float] = {
    "Bet365": 0.30,
    "WilliamHill": 0.27,
    "Pinnacle": 0.32,
    "Ladbrokes": 0.25,
    "Coral": 0.24,
    "SkyBet": 0.26,
    "Unibet": 0.26,
    "BoyleSports": 0.22,
    "Bwin": 0.25,
    "BetVictor": 0.24,
}


def _clip(x: float, lo: float = 0.10, hi: float = 0.60) -> float:
    return max(lo, min(hi, x))


class CompanyWeightService:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_table(self):
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS company_weights (
                    company TEXT PRIMARY KEY,
                    weight REAL NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def get(self, company: str) -> Optional[float]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT weight FROM company_weights WHERE company = ?",
                (company,),
            )
            row = cur.fetchone()
            if row:
                return float(row[0])
        # fallback to base
        return BASE_WEIGHTS.get(company)

    def upsert(self, company: str, weight: float, inc_samples: int = 1):
        weight = _clip(weight)
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT weight, sample_count FROM company_weights WHERE company = ?",
                (company,),
            )
            row = cur.fetchone()
            if row:
                prev_w, prev_n = float(row[0]), int(row[1])
                new_n = prev_n + inc_samples
                # EWMA 融合（越多样本越平滑）
                alpha = 0.30
                new_w = _clip(alpha * weight + (1 - alpha) * prev_w)
                conn.execute(
                    "UPDATE company_weights SET weight = ?, sample_count = ?, updated_at = ? WHERE company = ?",
                    (new_w, new_n, now, company),
                )
            else:
                conn.execute(
                    "INSERT INTO company_weights(company, weight, sample_count, updated_at) VALUES (?,?,?,?)",
                    (company, weight, inc_samples, now),
                )

    def batch_get(self, companies: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in companies:
            w = self.get(c)
            if w is None:
                # 未知公司给一个温和权重
                w = 0.24
            out[c] = _clip(float(w))
        return out

    # 依据盘口漂移与数据完整性，对公司权重做一次学习更新
    def learn_from_odds(self, odds_rows: List[dict]):
        # 预期输入：List[OddsData] 的类似字典，至少包含 company、initial_3way、final_3way 等可推导字段
        for o in odds_rows:
            company = o.get("company") or o.get("source") or "Unknown"
            # 计算 1X2 隐含概率的漂移幅度（缺数据则温和惩罚）
            try:
                ih, id, ia = o.get("initial_home_odds"), o.get("initial_draw_odds"), o.get("initial_away_odds")
                fh, fd, fa = o.get("final_home_odds"), o.get("final_draw_odds"), o.get("final_away_odds")
                def _imp(x):
                    return 1.0 / float(x) if x and float(x) > 0 else None
                ip = [_imp(ih), _imp(id), _imp(ia)]
                fp = [_imp(fh), _imp(fd), _imp(fa)]
                drift = 0.0
                cnt = 0
                for a, b in zip(ip, fp):
                    if a is not None and b is not None:
                        drift += abs(a - b)
                        cnt += 1
                drift = drift / cnt if cnt else 0.05  # 缺数据时给一个默认小漂移
                # 漂移越小，认为稳定性越高，略微上调权重；缺数据则下调
                base = self.get(company) or BASE_WEIGHTS.get(company, 0.24)
                quality_adj = 0.05 * (1.0 - min(0.5, drift) / 0.5)  # 0~+0.05
                missing_penalty = 0.04 if cnt < 3 else 0.0
                learned = _clip(base + quality_adj - missing_penalty)
                self.upsert(company, learned, inc_samples=1)
            except Exception:
                # 异常时温和惩罚
                base = self.get(company) or BASE_WEIGHTS.get(company, 0.24)
                learned = _clip(base - 0.03)
                self.upsert(company, learned, inc_samples=1)