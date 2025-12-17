import os
from typing import Optional, Dict, Any

import pandas as pd


class FormationStore:
    """
    读取并查询球队的阵型表现（来自 formation.xlsx）。
    - 支持按球队、赛季和阵型筛选；
    - 统一列名与别名；
    - 返回用于调整的核心指标：xg90、xga90、xg、xga、xgd、min。
    """

    def __init__(self, xlsx_path: str):
        self.xlsx_path = xlsx_path
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _resolve_path(self) -> str:
        path = self.xlsx_path
        if not os.path.isabs(path):
            base_dir = os.path.dirname(__file__)
            # 默认退回到仓库根目录下的 formation.xlsx
            path = os.path.abspath(os.path.join(base_dir, "..", "..", "..", "formation.xlsx"))
        return path

    def _load(self):
        path = self._resolve_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Formation Excel not found: {path}")
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception:
            # 允许 pandas 自动选择引擎的回退
            df = pd.read_excel(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        alias = {
            "team": "team",
            "球队": "team",
            "season": "season",
            "赛季": "season",
            "formation": "formation",
            "阵型": "formation",
            "min": "min",
            "minutes": "min",
            "sh": "sh",
            "shots": "sh",
            "g": "g",
            "goals": "g",
            "sha": "sha",
            "shots_against": "sha",
            "ga": "ga",
            "goals_against": "ga",
            "xg": "xg",
            "xga": "xga",
            "xgd": "xgd",
            "xg90": "xg90",
            "xga90": "xga90",
        }
        for k, v in alias.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        # 只保留核心列，其他列保留在 df 中以便扩展
        self._df = df

    @staticmethod
    def _norm_team(s: str) -> str:
        return (s or "").strip().lower()

    @staticmethod
    def _norm_formation(s: Optional[str]) -> Optional[str]:
        if not s:
            return None
        s = str(s).strip().lower()
        # 归一化到形如 4-2-3-1/4-3-3 等，仅保留数字与连接符
        import re
        parts = re.findall(r"\d+", s)
        return "-".join(parts) if parts else s

    def get_team_formation(self, team: str, formation: Optional[str] = None, season: Optional[str] = None) -> Optional[Dict[str, Any]]:
        df = self._df
        if df is None or len(df) == 0:
            return None
        team_norm = self._norm_team(team)
        q = df[df.get("team").astype(str).str.strip().str.lower() == team_norm]
        if season and "season" in df.columns:
            q = q[q.get("season").astype(str).str.strip() == str(season).strip()]
        if formation and "formation" in df.columns:
            f_norm = self._norm_formation(formation)
            # 对比归一化后的阵型
            def norm_series(x):
                try:
                    return self._norm_formation(x)
                except Exception:
                    return str(x)
            q = q[q.get("formation").apply(norm_series) == f_norm]

        if len(q) == 0:
            # 若按阵型与赛季均未命中，回退：该队伍分钟数最多的一条作为代表阵型
            q = df[df.get("team").astype(str).str.strip().str.lower() == team_norm]
            if len(q) == 0:
                return None
        # 选择分钟数最大的记录（代表主要使用阵型）
        if "min" in q.columns:
            q = q.sort_values(by="min", ascending=False)
        row = q.iloc[0].to_dict()

        # 构造输出（缺失时返回 None）
        def _num(x):
            try:
                return float(x)
            except Exception:
                return None
        res = {
            "team": row.get("team"),
            "season": row.get("season"),
            "formation": row.get("formation"),
            "min": _num(row.get("min")),
            "xg": _num(row.get("xg")),
            "xga": _num(row.get("xga")),
            "xgd": _num(row.get("xgd")),
            "xg90": _num(row.get("xg90")),
            "xga90": _num(row.get("xga90")),
            "g": _num(row.get("g")),
            "ga": _num(row.get("ga")),
            "sh": _num(row.get("sh")),
            "sha": _num(row.get("sha")),
        }
        return res