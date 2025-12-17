import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd


class HistoricalMarketEngine:
    """
    历史市场对照引擎：
    - 从 SQLite 历史库中抽取胜平负初/终盘赔率（四家主流公司：365、皇冠、澳门、易胜博）
    - 构建历史画像（终盘隐含概率均值、从初盘到终盘的平均变化）
    - 与当前赔率进行对照，生成中文摘要用于报告
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

    def _normalize_team_name(self, name: str) -> str:
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
            "谢菲联": "谢菲尔德联",
            "西汉姆联": "西汉姆",
            "狼队": "狼队",
        }
        return aliases.get(s.replace(" ", ""), s.replace(" ", ""))

    # —— 赔率工具 ——
    def _implied_from_triple(self, h: float, d: float, a: float) -> Tuple[float, float, float]:
        margin = (1 / h) + (1 / d) + (1 / a)
        return (1 / h) / margin, (1 / d) / margin, (1 / a) / margin

    def _company_columns(self, cols: set) -> Dict[str, Dict[str, Optional[str]]]:
        """返回各公司的胜平负列映射（初盘、终盘）。
        兼容 matches_data 的中文列命名模式：
        - 365: 365_初赔, 365, 365_2, 365_终赔, 365_3, 365_4
        - 皇冠（crown）: 皇冠（crown）_初赔, 皇冠（crown）, 皇冠（crown）_2, 皇冠（crown）_终赔, 皇冠（crown）_3, 皇冠（crown）_4
        - 澳门: 澳门_初赔, 澳门, 澳门_2, 澳门_终赔, 澳门_3, 澳门_4
        - 易胜博: 易胜博_初赔, 易胜博, 易胜博_2, 易胜博_终赔, 易胜博_3, 易胜博_4
        """
        def mk(prefix: str) -> Dict[str, Optional[str]]:
            return {
                "initial_home": self._pick_col(cols, [f"{prefix}_初赔"]),
                "initial_draw": self._pick_col(cols, [prefix]),
                "initial_away": self._pick_col(cols, [f"{prefix}_2"]),
                "final_home": self._pick_col(cols, [f"{prefix}_终赔"]),
                "final_draw": self._pick_col(cols, [f"{prefix}_3"]),
                "final_away": self._pick_col(cols, [f"{prefix}_4"]),
            }

        mapping = {
            "365": mk("365"),
            "皇冠（crown）": mk("皇冠（crown）"),
            "澳门": mk("澳门"),
            "易胜博": mk("易胜博"),
        }
        # 仅保留存在初终盘主胜列的公司
        return {k: v for k, v in mapping.items() if v["initial_home"] and v["final_home"]}

    def _load_df(self, conn: sqlite3.Connection, table: str) -> pd.DataFrame:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.columns = [c.strip() for c in df.columns]
        return df

    def _filter_recent(self, df: pd.DataFrame) -> pd.DataFrame:
        season_col = self._pick_col(set(df.columns), ["比赛信息", "season", "赛季", "Season", "year"])
        if season_col is None:
            return df
        def season_year(x) -> Optional[int]:
            try:
                s = str(x)
                if s.strip() == "" or s.strip() in ("赛季",):
                    return None
                head = s.split("-")[0]
                return int(head)
            except Exception:
                return None
        df["__season_year"] = df[season_col].apply(season_year)
        return df[(df["__season_year"].notna()) & (df["__season_year"] >= 2018)].copy()

    def _filter_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        # 剔除 matches_data 中的标题行（例如 基本面_1 == "主队"）
        base_col = self._pick_col(set(df.columns), ["基本面_1", "主队", "home_team", "HomeTeam"])
        if base_col and base_col in df.columns:
            return df[df[base_col] != "主队"].copy()
        return df

    def _resolve_league(self, df: pd.DataFrame, home_name: str, away_name: str) -> Optional[str]:
        cols = set(df.columns)
        league_col = self._pick_col(cols, ["联赛", "league", "赛事", "unnamed_2"])  # 观测到 unnamed_2 为联赛
        home_col = self._pick_col(cols, ["基本面_2", "home_team", "主队"])
        away_col = self._pick_col(cols, ["基本面_9", "away_team", "客队"])
        if not league_col or not home_col or not away_col:
            return None
        hn = self._normalize_team_name(home_name)
        an = self._normalize_team_name(away_name)
        tmp = df.copy()
        tmp["__home_n"] = tmp[home_col].astype(str).apply(self._normalize_team_name)
        tmp["__away_n"] = tmp[away_col].astype(str).apply(self._normalize_team_name)
        cand = tmp[(tmp["__home_n"] == hn) | (tmp["__away_n"] == an)]
        if cand.empty:
            return None
        # 取出现频率最高的联赛作为当前匹配的联赛
        return cand[league_col].astype(str).mode().iloc[0]

    # —— 主流程 ——
    def analyze_historical_comparison(self, home_name: str, away_name: str, odds_data: List[Dict]) -> str:
        """构建历史画像并与当前赔率进行对照，返回中文摘要。"""
        try:
            conn = self._connect()
            table = self._find_table(conn)
            if not table:
                return "（历史库中未找到比赛表，无法进行市场对照。）"
            df = self._load_df(conn, table)
        except Exception:
            return "（历史库连接失败或数据缺失，无法进行市场对照。）"
        finally:
            try:
                conn.close()
            except Exception:
                pass

        # 基本清洗
        df = self._filter_headers(df)
        df = self._filter_recent(df)

        # 识别公司列
        companies = self._company_columns(set(df.columns))
        if not companies:
            return "（历史库缺少主流公司胜平负赔率列，无法进行市场对照。）"

        # 若能解析当前比赛联赛，则按联赛过滤；否则使用全库
        league = self._resolve_league(df, home_name, away_name)
        if league:
            df_league = df[df[self._pick_col(set(df.columns), ["联赛", "league", "赛事", "unnamed_2"])].astype(str) == str(league)].copy()
        else:
            df_league = df

        # 遍历历史行，计算各公司初/终盘隐含概率
        hist_final_triples: List[Tuple[float, float, float]] = []
        hist_initial_triples: List[Tuple[float, float, float]] = []
        for _, row in df_league.iterrows():
            for comp, cols in companies.items():
                try:
                    ih, idr, ia = float(row[cols["initial_home"]]), float(row[cols["initial_draw"]]), float(row[cols["initial_away"]])
                    fh, fdr, fa = float(row[cols["final_home"]]), float(row[cols["final_draw"]]), float(row[cols["final_away"]])
                except Exception:
                    continue
                # 过滤不合理赔率
                if min(ih, idr, ia, fh, fdr, fa) <= 1e-6:
                    continue
                initial_probs = self._implied_from_triple(ih, idr, ia)
                final_probs = self._implied_from_triple(fh, fdr, fa)
                hist_initial_triples.append(initial_probs)
                hist_final_triples.append(final_probs)

        if not hist_final_triples or not hist_initial_triples:
            return "（历史样本中缺少有效赔率数据，无法进行市场对照。）"

        # 历史均值（聚合四家公司）
        import numpy as np
        hist_final_avg = tuple(np.mean(np.array(hist_final_triples), axis=0))
        hist_initial_avg = tuple(np.mean(np.array(hist_initial_triples), axis=0))
        hist_drift = tuple(f - i for f, i in zip(hist_final_avg, hist_initial_avg))

        # 当前赔率聚合（使用传入公司，不限定四家）
        cur_initial_triples = []
        cur_final_triples = []
        for o in odds_data:
            try:
                ih, idr, ia = float(o["initial_home_win"]), float(o["initial_draw"]), float(o["initial_away_win"])  # type: ignore
                fh, fdr, fa = float(o["final_home_win"]), float(o["final_draw"]), float(o["final_away_win"])  # type: ignore
            except Exception:
                continue
            cur_initial_triples.append(self._implied_from_triple(ih, idr, ia))
            cur_final_triples.append(self._implied_from_triple(fh, fdr, fa))

        if not cur_final_triples or not cur_initial_triples:
            return "（当前请求缺少有效的胜平负赔率数据，无法进行市场对照。）"

        cur_final_avg = tuple(np.mean(np.array(cur_final_triples), axis=0))
        cur_initial_avg = tuple(np.mean(np.array(cur_initial_triples), axis=0))
        cur_drift = tuple(f - i for f, i in zip(cur_final_avg, cur_initial_avg))

        # 生成摘要
        def pct3(x: float) -> str:
            return f"{x*100:.1f}%"

        summary = (
            f"历史（{'同联赛' if league else '全库'}）终盘隐含概率均值：主胜{pct3(hist_final_avg[0])}、平局{pct3(hist_final_avg[1])}、客胜{pct3(hist_final_avg[2])}。\n"
            f"历史平均变化（初→终）：主胜{pct3(hist_drift[0])}、平局{pct3(hist_drift[1])}、客胜{pct3(hist_drift[2])}。\n"
            f"当前市场终盘均值：主胜{pct3(cur_final_avg[0])}、平局{pct3(cur_final_avg[1])}、客胜{pct3(cur_final_avg[2])}；变化：主胜{pct3(cur_drift[0])}、平局{pct3(cur_drift[1])}、客胜{pct3(cur_drift[2])}。\n"
        )

        # 简要判断方向
        def dir_word(delta: float) -> str:
            if abs(delta) < 0.01:
                return "基本一致"
            return "更高" if delta > 0 else "更低"

        summary += (
            f"对照结论：当前市场主胜终盘隐含概率{dir_word(cur_final_avg[0]-hist_final_avg[0])}于历史均值；"
            f"平局{dir_word(cur_final_avg[1]-hist_final_avg[1])}；"
            f"客胜{dir_word(cur_final_avg[2]-hist_final_avg[2])}。"
        )

        return summary

    def list_companies(self) -> List[str]:
        """返回数据库中识别到的主流博彩公司名称列表，用于前端下拉同步。"""
        try:
            conn = self._connect()
            table = self._find_table(conn)
            if not table:
                return []
            df = self._load_df(conn, table)
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

        df = self._filter_headers(df)
        df = self._filter_recent(df)
        companies = self._company_columns(set(df.columns))
        return list(companies.keys())