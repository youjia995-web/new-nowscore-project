import math
from typing import Dict, Any, List, Optional

try:
    from .calibration import CalibrationService
except Exception:
    CalibrationService = None


class ProbCalibrationService:
    """
    基于历史预测与真实结果的后验概率校准服务。
    运行时仅做插值与归一化，不依赖 scikit-learn（训练脚本负责写入参数）。

    期望 calibration_params 表中存在 key=prob_calibration_v1，结构：
    {
      "points": {
        "home": [[p_in, p_out], ...],
        "draw": [[p_in, p_out], ...],
        "away": [[p_in, p_out], ...]
      },
      "meta": {"bin_count": 51, "source": "prediction_logs"}
    }
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.available = False
        self.payload: Optional[Dict[str, Any]] = None
        self.points: Dict[str, List[List[float]]] = {}
        if CalibrationService:
            try:
                cs = CalibrationService(db_path)
                self.payload = cs.get("prob_calibration_v1")
                pts = (self.payload or {}).get("points") or {}
                # 只接受合理的点集
                for k in ("home", "draw", "away"):
                    v = pts.get(k) or []
                    if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                        # 保证按输入值升序
                        self.points[k] = sorted([[float(a), float(b)] for a, b in v], key=lambda t: t[0])
                self.available = all(self.points.get(k) for k in ("home", "draw", "away"))
            except Exception:
                self.available = False

    @staticmethod
    def _interp(x: float, pts: List[List[float]]) -> float:
        if not pts:
            return x
        # 边界裁剪
        if x <= pts[0][0]:
            return max(0.0, min(1.0, float(pts[0][1])))
        if x >= pts[-1][0]:
            return max(0.0, min(1.0, float(pts[-1][1])))
        # 线性插值
        for i in range(1, len(pts)):
            x1 = pts[i][0]
            if x <= x1:
                x0, y0 = pts[i - 1]
                x1, y1 = pts[i]
                if x1 == x0:
                    return max(0.0, min(1.0, float(y1)))
                t = (x - x0) / (x1 - x0)
                y = y0 + t * (y1 - y0)
                return max(0.0, min(1.0, float(y)))
        return max(0.0, min(1.0, x))

    def apply(self, dist: Dict[str, float]) -> Dict[str, float]:
        """
        对三路概率进行单调插值校准，并归一化。
        输入/输出键：home_win、draw、away_win。
        """
        h = float(dist.get("home_win", 0.0))
        d = float(dist.get("draw", 0.0))
        a = float(dist.get("away_win", 0.0))
        if not self.available:
            s = h + d + a
            if s <= 0:
                return {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
            return {"home_win": h / s, "draw": d / s, "away_win": a / s}

        ch = self._interp(h, self.points.get("home") or [])
        cd = self._interp(d, self.points.get("draw") or [])
        ca = self._interp(a, self.points.get("away") or [])
        s = ch + cd + ca
        if s <= 1e-12:
            # 极端情况下保持原分布
            s0 = h + d + a
            if s0 <= 0:
                return {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
            return {"home_win": h / s0, "draw": d / s0, "away_win": a / s0}
        return {"home_win": ch / s, "draw": cd / s, "away_win": ca / s}