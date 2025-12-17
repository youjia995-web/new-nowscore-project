import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class HistoricalStatsEngine:
    """
    历史市场结构化统计引擎：
    - 从 SQLite 历史库抽取主流公司胜平负初/终盘赔率
    - 计算历史终盘隐含概率均值、分位点（25/50/75）
    - 计算当前公司的终盘与初盘均值及变化
    - 返回结构化字典供前端图表与稳定性提示使用
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    # —— 基础工具 ——
    def _connect(self) -> sqlite3.Connection:
        path = self.db_path
        if not os.path.isabs(path):
            base_dir = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(base_dir, path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"DB not found: {path}")
        return sqlite3.connect(path)

    def _find_table(self, conn: sqlite3.Connection) -> Optional[str]:
        candidates = ["matches_data", "matches", "results", "fixtures", "games"]
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0].lower(): row[0] for row in cur.fetchall()}
        for c in candidates:
            if c in tables:
                return tables[c]
        for low, orig in tables.items():
            if not low.startswith("sqlite_"):
                return orig
        return None

    def _pick_col(self, cols: set, candidates) -> Optional[str]:
        lower_map = {str(c).strip().lower(): c for c in cols}
        for c in candidates:
            lc = str(c).strip().lower()
            if lc in lower_map:
                return lower_map[lc]
        for c in candidates:
            lc = str(c).strip().lower()
            for col in cols:
                col_name = str(col).strip().lower()
                if lc and lc in col_name:
                    return col
        return None

    def _filter_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        # 过滤标题行（常见中文标题或英文标题）
        cols = set(df.columns)
        home_col = self._pick_col(cols, ["主队", "home", "home_team", "主队名称", "主队名"])
        away_col = self._pick_col(cols, ["客队", "away", "away_team", "客队名称", "客队名"])
        if not home_col or not away_col:
            return df
        mask = ~(
            df[home_col].astype(str).str.contains("主队|Home", na=False) |
            df[away_col].astype(str).str.contains("客队|Away", na=False)
        )
        return df[mask].copy()

    def _filter_recent(self, df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
        # 仅保留近 N 年赛季或日期
        cols = set(df.columns)
        season_col = self._pick_col(cols, ["season", "Season", "赛季", "比赛信息"])  # matches_data 中赛季在“比赛信息”列
        date_col = self._pick_col(cols, ["date", "Date", "比赛日期", "datetime", "match_time"])  # 可选
        if season_col is not None:
            # 赛季格式如 2023-2024 或 2023
            def parse_year(s):
                try:
                    s = str(s)
                    for token in s.replace("_", "-").split("-"):
                        if token.isdigit():
                            y = int(token)
                            if 1900 < y < 2100:
                                return y
                except Exception:
                    return None
                return None
            df["_year"] = df[season_col].apply(parse_year)
            max_year = df["_year"].dropna().max() if df["_year"].dropna().size > 0 else None
            if max_year is not None:
                return df[(df["_year"] >= max_year - years)].copy()
        return df

    def _company_columns(self, cols: set) -> Dict[str, Dict[str, str]]:
        # 识别四家主流公司列（英文/中文/别名）
        companies = {
            "bet365": ["bet365", "365", "博365", "博弈365"],
            "crown": ["crown", "皇冠", "Crown"],
            "macau": ["macau", "澳门", "Macau", "澳彩"],
            "ysb": ["ysb", "易胜博", "Ysb", "YingShengBo"],
        }
        patterns = {
            "initial_home": ["初盘主胜", "初盘主胜赔率", "initial_home_win", "初盘主胜(欧指)", "初盘主胜_欧"] ,
            "initial_draw": ["初盘平局", "初盘平局赔率", "initial_draw", "初盘平局(欧指)", "初盘平局_欧"],
            "initial_away": ["初盘客胜", "初盘客胜赔率", "initial_away_win", "初盘客胜(欧指)", "初盘客胜_欧"],
            "final_home": ["终盘主胜", "终盘主胜赔率", "final_home_win", "终盘主胜(欧指)", "终盘主胜_欧"],
            "final_draw": ["终盘平局", "终盘平局赔率", "final_draw", "终盘平局(欧指)", "终盘平局_欧"],
            "final_away": ["终盘客胜", "终盘客胜赔率", "final_away_win", "终盘客胜(欧指)", "终盘客胜_欧"],
        }
        result: Dict[str, Dict[str, str]] = {}
        lower_cols = {str(c).strip().lower(): c for c in cols}
        for comp, aliases in companies.items():
            comp_cols = {}
            for key, pats in patterns.items():
                found = None
                for p in pats:
                    for lc, orig in lower_cols.items():
                        if p.lower() in lc:
                            found = orig
                            break
                    if found:
                        break
                comp_cols[key] = found
            if all(comp_cols.values()):
                result[comp] = comp_cols
        return result

    def _implied_from_triple(self, h: float, d: float, a: float) -> Tuple[float, float, float]:
        inv = np.array([1.0 / h, 1.0 / d, 1.0 / a])
        total = inv.sum()
        probs = inv / total
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _resolve_league(self, df: pd.DataFrame, home_name: str, away_name: str) -> Optional[str]:
        # 从一行包含两队的行获取联赛（如果能找到）
        cols = set(df.columns)
        home_col = self._pick_col(cols, ["主队", "home", "home_team", "主队名称", "主队名"])
        away_col = self._pick_col(cols, ["客队", "away", "away_team", "客队名称", "客队名"])
        league_col = self._pick_col(cols, ["联赛", "league", "赛事", "unnamed_2"])
        if not (home_col and away_col and league_col):
            return None

        def norm(name: Optional[str]) -> str:
            if name is None:
                return ""
            s = str(name).strip()
            try:
                import re
                s = re.sub(r"\[.*?\]", "", s)
                s = re.sub(r"\(.*?\)", "", s)
            except Exception:
                pass
            aliases = {
                "曼城": "曼彻斯特城",
                "曼联": "曼彻斯特联",
                "纽卡": "纽卡斯尔联",
                "纽卡斯尔": "纽卡斯尔联",
            }
            return aliases.get(s, s)

        h = norm(home_name)
        a = norm(away_name)
        sample = df[(df[home_col].astype(str) == h) & (df[away_col].astype(str) == a)]
        if not sample.empty:
            return str(sample.iloc[0][league_col])
        return None

    def _load_df(self, conn: sqlite3.Connection, table: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cur.fetchall()]
            return pd.DataFrame(columns=cols)

    # —— 结构化统计主流程 ——
    def compute(self, home_name: str, away_name: str, odds_data: List[Dict]) -> Dict:
        """
        返回结构化历史市场统计：均值、分位点、当前均值与变化。
        """
        try:
            conn = self._connect()
            table = self._find_table(conn)
            if not table:
                return {"error": "历史库中未找到比赛表"}
            df = self._load_df(conn, table)
        except Exception:
            return {"error": "历史库连接失败或数据缺失"}
        finally:
            try:
                conn.close()
            except Exception:
                pass

        df = self._filter_headers(df)
        df = self._filter_recent(df)

        companies = self._company_columns(set(df.columns))
        if not companies:
            return {"error": "缺少主流公司胜平负赔率列"}

        league = self._resolve_league(df, home_name, away_name)
        if league:
            league_col = self._pick_col(set(df.columns), ["联赛", "league", "赛事", "unnamed_2"])
            df_league = df[df[league_col].astype(str) == str(league)].copy()
        else:
            df_league = df

        hist_final_triples: List[Tuple[float, float, float]] = []
        hist_initial_triples: List[Tuple[float, float, float]] = []
        for _, row in df_league.iterrows():
            for _, cols in companies.items():
                try:
                    ih, idr, ia = float(row[cols["initial_home"]]), float(row[cols["initial_draw"]]), float(row[cols["initial_away"]])
                    fh, fdr, fa = float(row[cols["final_home"]]), float(row[cols["final_draw"]]), float(row[cols["final_away"]])
                except Exception:
                    continue
                if min(ih, idr, ia, fh, fdr, fa) <= 1e-6:
                    continue
                hist_initial_triples.append(self._implied_from_triple(ih, idr, ia))
                hist_final_triples.append(self._implied_from_triple(fh, fdr, fa))

        if not hist_final_triples:
            return {"error": "历史样本不足"}

        hist_final_arr = np.array(hist_final_triples)
        hist_initial_arr = np.array(hist_initial_triples) if hist_initial_triples else None

        hist_final_avg = tuple(np.mean(hist_final_arr, axis=0))
        hist_initial_avg = tuple(np.mean(hist_initial_arr, axis=0)) if hist_initial_arr is not None else (0.0, 0.0, 0.0)

        def percentiles(arr: np.ndarray, idx: int) -> Dict[str, float]:
            col = arr[:, idx]
            return {
                "p25": float(np.percentile(col, 25)),
                "p50": float(np.percentile(col, 50)),
                "p75": float(np.percentile(col, 75)),
            }

        hist_percentiles = {
            "home": percentiles(hist_final_arr, 0),
            "draw": percentiles(hist_final_arr, 1),
            "away": percentiles(hist_final_arr, 2),
        }

        # 当前均值（使用传入公司集合）
        cur_initial_triples = []
        cur_final_triples = []
        for o in odds_data:
            try:
                ih, idr, ia = float(o["initial_home_win"]), float(o["initial_draw"]), float(o["initial_away_win"])  # type: ignore
                fh, fdr, fa = float(o["final_home_win"]), float(o["final_draw"]), float(o["final_away_win"])  # type: ignore
            except Exception:
                continue
            if min(ih, idr, ia, fh, fdr, fa) <= 1e-6:
                continue
            cur_initial_triples.append(self._implied_from_triple(ih, idr, ia))
            cur_final_triples.append(self._implied_from_triple(fh, fdr, fa))

        if cur_final_triples:
            cur_final_avg = tuple(np.mean(np.array(cur_final_triples), axis=0))
        else:
            cur_final_avg = (0.0, 0.0, 0.0)
        if cur_initial_triples:
            cur_initial_avg = tuple(np.mean(np.array(cur_initial_triples), axis=0))
        else:
            cur_initial_avg = (0.0, 0.0, 0.0)
        cur_drift = tuple(f - i for f, i in zip(cur_final_avg, cur_initial_avg))

        return {
            "scope": "同联赛" if league else "全库",
            "hist_final_avg": {"home": hist_final_avg[0], "draw": hist_final_avg[1], "away": hist_final_avg[2]},
            "hist_initial_avg": {"home": hist_initial_avg[0], "draw": hist_initial_avg[1], "away": hist_initial_avg[2]},
            "hist_percentiles": hist_percentiles,
            "cur_final_avg": {"home": cur_final_avg[0], "draw": cur_final_avg[1], "away": cur_final_avg[2]},
            "cur_initial_avg": {"home": cur_initial_avg[0], "draw": cur_initial_avg[1], "away": cur_initial_avg[2]},
            "cur_drift": {"home": cur_drift[0], "draw": cur_drift[1], "away": cur_drift[2]},
        }