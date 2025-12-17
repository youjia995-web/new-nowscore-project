import os
import sqlite3
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


class HistoryEngine:
    """历史数据引擎
    - 从 SQLite 数据库生成可用于校正模型概率的历史信号
    - 产出用于 AI 报告的中文摘要
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        path = self.db_path
        if not os.path.isabs(path):
            # 将相对路径视为相对于当前文件所在目录的路径
            base_dir = os.path.dirname(__file__)
            path = os.path.abspath(os.path.join(base_dir, path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"DB not found: {path}")
        return sqlite3.connect(path)

    def _find_table(self, conn: sqlite3.Connection) -> Optional[str]:
        """尝试匹配常见比赛表名，优先选择包含完整文本字段的表。
        优先级：matches_data > matches > results > fixtures > games
        找不到则返回第一张非系统表
        """
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
        """在列集合中查找候选字段，支持大小写不敏感与模糊匹配。
        - 先执行大小写不敏感的精确匹配
        - 若失败，再执行包含关系的模糊匹配（例如 'home' 匹配 'home_team_name'）
        """
        lower_map = {str(c).strip().lower(): c for c in cols}
        # 精确（不区分大小写）
        for c in candidates:
            lc = str(c).strip().lower()
            if lc in lower_map:
                return lower_map[lc]
        # 模糊包含
        for c in candidates:
            lc = str(c).strip().lower()
            for col in cols:
                col_name = str(col).strip().lower()
                if not lc:
                    continue
                # 避免误匹配到 *_id / team_id 等纯ID字段
                if (lc in ("home", "away") and "team_id" in col_name) or col_name.endswith("_id") or col_name.endswith("id"):
                    continue
                if lc in col_name:
                    return col
        return None

    def _load_matches_df(self, conn: sqlite3.Connection, table: str) -> pd.DataFrame:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.columns = [c.strip() for c in df.columns]
        return df

    def _resolve_schema(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = set(df.columns)
        home_team = self._pick_col(
            cols,
            [
                "home_team",
                "HomeTeam",
                "home",
                "home_name",
                "team_home",
                "homeTeam",
                "主队",
                "主队名称",
                # matches_data 观测到的主队名称列
                "基本面_2",
            ],
        )
        away_team = self._pick_col(
            cols,
            [
                "away_team",
                "AwayTeam",
                "away",
                "away_name",
                "team_away",
                "awayTeam",
                "客队",
                "客队名称",
                # matches_data 观测到的客队名称列
                "基本面_9",
            ],
        )
        home_goals = self._pick_col(
            cols,
            [
                "home_goals",
                "HomeGoals",
                "FTHG",
                "home_score",
                "goals_home",
                "主队进球",
                "主队得分",
                # matches_data 观测到的全场主队进球列
                "半全场数据_6",
            ],
        )
        away_goals = self._pick_col(
            cols,
            [
                "away_goals",
                "AwayGoals",
                "FTAG",
                "away_score",
                "goals_away",
                "客队进球",
                "客队得分",
                # matches_data 观测到的全场客队进球列
                "半全场数据_7",
            ],
        )
        result_col = self._pick_col(cols, ["result", "FTR", "赛果", "全场结果", "半全场数据_全场赛果"])  # FTR: H/D/A 或 中文
        season_col = self._pick_col(cols, ["season", "Season", "year", "赛季", "比赛信息"])  # matches_data 中赛季在“比赛信息”列
        date_col = self._pick_col(cols, ["date", "Date", "match_date", "比赛日期", "datetime", "match_time"])
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result_col,
            "season": season_col,
            "date": date_col,
        }

    def _resolve_team_names_via_ids(self, conn: sqlite3.Connection, df: pd.DataFrame, sch: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        """当仅能识别到 *_team_id 字段时，尝试通过 teams 表映射到名称。"""
        # 若已识别到名称列，直接返回
        if sch.get("home_team") and sch.get("away_team"):
            return sch

        cols = set(df.columns)
        home_id_col = self._pick_col(cols, ["home_team_id", "home_id", "主队ID"]) 
        away_id_col = self._pick_col(cols, ["away_team_id", "away_id", "客队ID"]) 
        if not home_id_col or not away_id_col:
            return sch

        # 尝试加载 teams 表
        try:
            teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
            # 识别 id 与 name 列
            teams_id_col = None
            teams_name_col = None
            tcols = set(teams_df.columns)
            teams_id_col = self._pick_col(tcols, ["id", "team_id", "编号", "ID"]) or list(teams_df.columns)[0]
            teams_name_col = self._pick_col(tcols, ["name", "team_name", "名称", "队名"]) or (list(teams_df.columns)[1] if len(teams_df.columns) > 1 else None)
            if not teams_id_col or not teams_name_col:
                return sch

            # 构建映射并生成名称列
            id_to_name = dict(zip(teams_df[teams_id_col], teams_df[teams_name_col]))
            df["home_team_resolved"] = df[home_id_col].map(id_to_name)
            df["away_team_resolved"] = df[away_id_col].map(id_to_name)
            sch["home_team"] = "home_team_resolved"
            sch["away_team"] = "away_team_resolved"
            return sch
        except Exception:
            return sch

    def _compute_outcome(self, row, sch) -> str:
        """输出 H/D/A（主胜/平/客胜）"""
        if sch["result"] and pd.notna(row[sch["result"]]):
            val_raw = str(row[sch["result"]]).strip()
            val = val_raw.upper()
            # 英文编码 H/D/A
            if val in ("H", "D", "A"):
                return val
            # 中文编码 胜/平/负 -> H/D/A
            if val_raw in ("胜", "主胜"):
                return "H"
            if val_raw in ("平", "平局"):
                return "D"
            if val_raw in ("负", "客胜"):
                return "A"
        if (
            sch["home_goals"]
            and sch["away_goals"]
            and pd.notna(row[sch["home_goals"]])
            and pd.notna(row[sch["away_goals"]])
        ):
            if row[sch["home_goals"]] > row[sch["away_goals"]]:
                return "H"
            elif row[sch["home_goals"]] < row[sch["away_goals"]]:
                return "A"
            else:
                return "D"
        return "D"

    def _normalize_team_name(self, name: str) -> str:
        """统一球队名称（中文常见别名、去括号标注、去空格）。"""
        if name is None:
            return ""
        s = str(name).strip()
        # 去除如 [英超-11] 等括号标注
        try:
            import re
            s = re.sub(r"\[.*?\]", "", s)
            s = re.sub(r"\(.*?\)", "", s)
        except Exception:
            pass
        # 常见中文别名归一
        aliases = {
            "曼城": "曼彻斯特城",
            "曼联": "曼彻斯特联",
            "纽卡": "纽卡斯尔联",
            "纽卡斯尔": "纽卡斯尔联",
            "谢菲联": "谢菲尔德联",
            "西汉姆联": "西汉姆",
            "狼队": "狼队",  # 伍尔弗汉普顿流浪者 -> 狼队
        }
        s2 = s.replace(" ", "")
        return aliases.get(s2, s2)

    def _ppg(self, outcome: str, is_home: bool) -> float:
        """主场或客场每场积分：胜3分，平1分，负0分"""
        if outcome == "D":
            return 1.0
        if is_home and outcome == "H":
            return 3.0
        if (not is_home) and outcome == "A":
            return 3.0
        return 0.0

    def generate_signals(self, home_name: str, away_name: str) -> Tuple[Dict[str, float], str]:
        """产出历史信号和摘要。
        返回:
            signals: dict 包含用于校正的指标
            summary: 供 AI 报告的中文摘要
        """
        try:
            conn = self._connect()
            table = self._find_table(conn)
            if not table:
                return self._empty_signals("未找到比赛数据表")

            df = self._load_matches_df(conn, table)
            sch = self._resolve_schema(df)
            # 若识别到的是ID字段或未识别到名称，尝试通过teams表映射
            sch = self._resolve_team_names_via_ids(conn, df, sch)
            required = ["home_team", "away_team"]
            if any(sch[k] is None for k in required):
                return self._empty_signals("比赛表缺少主客队字段")

            # 过滤 2018 年及之后数据（若能识别 season 或 date）
            if sch["season"] and sch["season"] in df.columns:
                try:
                    season_str = df[sch["season"]].astype(str).str.strip()
                    # 去除非赛季值（例如首行包含“赛季”文本）
                    mask_valid = season_str.str.match(r"^\d{4}")
                    df = df[mask_valid]
                    df = df[season_str.str[:4].astype(int) >= 2018]
                except Exception:
                    pass
            elif sch["date"] and sch["date"] in df.columns:
                try:
                    year = pd.to_datetime(df[sch["date"]], errors="coerce").dt.year
                    df = df[year >= 2018]
                except Exception:
                    pass

            # 统一球队名称为归一化形式用于匹配
            df["home_team_norm"] = df[sch["home_team"]].astype(str).apply(self._normalize_team_name)
            df["away_team_norm"] = df[sch["away_team"]].astype(str).apply(self._normalize_team_name)
            home_name_norm = self._normalize_team_name(home_name)
            away_name_norm = self._normalize_team_name(away_name)

            # 进球数字列转换为数值型
            if sch["home_goals"] and sch["home_goals"] in df.columns:
                df[sch["home_goals"]] = pd.to_numeric(df[sch["home_goals"]], errors="coerce")
            if sch["away_goals"] and sch["away_goals"] in df.columns:
                df[sch["away_goals"]] = pd.to_numeric(df[sch["away_goals"]], errors="coerce")

            # 计算 H2H（互相对赛）
            h2h = df[
                (
                    (df["home_team_norm"] == home_name_norm)
                    & (df["away_team_norm"] == away_name_norm)
                )
                | (
                    (df["home_team_norm"] == away_name_norm)
                    & (df["away_team_norm"] == home_name_norm)
                )
            ].copy()

            if not h2h.empty:
                h2h["outcome"] = h2h.apply(lambda r: self._compute_outcome(r, sch), axis=1)
                h_total = len(h2h)
                h_home_wins = (
                    (h2h[sch["home_team"]] == home_name) & (h2h["outcome"] == "H")
                ).sum()
                h_away_wins = (
                    (h2h[sch["home_team"]] == away_name) & (h2h["outcome"] == "H")
                ).sum()
                h_draws = (h2h["outcome"] == "D").sum()
                h2h_home_win_rate = h_home_wins / h_total
                h2h_away_win_rate = h_away_wins / h_total
                h2h_draw_rate = h_draws / h_total
            else:
                h2h_home_win_rate = h2h_away_win_rate = h2h_draw_rate = 0.0

            # 主队主场 PPG、净胜球
            home_home = df[df["home_team_norm"] == home_name_norm].copy()
            if not home_home.empty:
                home_home["outcome"] = home_home.apply(
                    lambda r: self._compute_outcome(r, sch), axis=1
                )
                home_ppg = home_home["outcome"].apply(lambda o: self._ppg(o, is_home=True)).mean()
                if sch["home_goals"] and sch["away_goals"]:
                    home_goal_diff_avg = (
                        home_home[sch["home_goals"]] - home_home[sch["away_goals"]]
                    ).mean()
                else:
                    home_goal_diff_avg = 0.0
            else:
                home_ppg = 0.0
                home_goal_diff_avg = 0.0

            # 客队客场 PPG、净胜球
            away_away = df[df["away_team_norm"] == away_name_norm].copy()
            if not away_away.empty:
                away_away["outcome"] = away_away.apply(
                    lambda r: self._compute_outcome(r, sch), axis=1
                )
                away_ppg = away_away["outcome"].apply(lambda o: self._ppg(o, is_home=False)).mean()
                if sch["home_goals"] and sch["away_goals"]:
                    away_goal_diff_avg = (
                        away_away[sch["away_goals"]] - away_away[sch["home_goals"]]
                    ).mean()
                else:
                    away_goal_diff_avg = 0.0
            else:
                away_ppg = 0.0
                away_goal_diff_avg = 0.0

            # 组装信号
            signals = {
                "h2h_home_win_rate": float(h2h_home_win_rate),
                "h2h_draw_rate": float(h2h_draw_rate),
                "h2h_away_win_rate": float(h2h_away_win_rate),
                "home_ppg": float(home_ppg),
                "away_ppg": float(away_ppg),
                "home_goal_diff_avg": float(home_goal_diff_avg),
                "away_goal_diff_avg": float(away_goal_diff_avg),
            }

            summary = (
                f"历史数据摘要（2018–至今）：\n"
                f"- 互相对赛: 主胜{signals['h2h_home_win_rate']:.1%}，平{signals['h2h_draw_rate']:.1%}，客胜{signals['h2h_away_win_rate']:.1%}\n"
                f"- 主队主场PPG: {signals['home_ppg']:.2f}，客队客场PPG: {signals['away_ppg']:.2f}\n"
                f"- 主场平均净胜球: {signals['home_goal_diff_avg']:.2f}，客场平均净胜球: {signals['away_goal_diff_avg']:.2f}"
            )

            return signals, summary

        except Exception as e:
            return self._empty_signals(f"历史引擎异常: {e}")

    def _empty_signals(self, reason: str) -> Tuple[Dict[str, float], str]:
        signals = {
            "h2h_home_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_away_win_rate": 0.0,
            "home_ppg": 0.0,
            "away_ppg": 0.0,
            "home_goal_diff_avg": 0.0,
            "away_goal_diff_avg": 0.0,
        }
        summary = f"历史数据摘要：{reason}，不进行概率校正。"
        return signals, summary

    def apply_probability_adjustments(
        self, model_probs: Dict[str, float], signals: Dict[str, float]
    ) -> Dict[str, float]:
        """根据历史信号对胜平负概率做不超过±10%的微调，并归一化。"""
        # 表现差异（归一化成 [-1, 1]）
        form_diff = (signals["home_ppg"] - signals["away_ppg"]) / 3.0  # 每场最高 3 分
        h2h_bias = signals["h2h_home_win_rate"] - signals["h2h_away_win_rate"]
        goal_diff_factor = np.tanh(
            signals["home_goal_diff_avg"] - signals["away_goal_diff_avg"]
        )

        # 综合调整（权重可根据实际效果微调）
        raw_adj = (form_diff * 0.08) + (h2h_bias * 0.05) + (goal_diff_factor * 0.04)
        adj = float(np.clip(raw_adj, -0.10, 0.10))

        home = model_probs["home_win"] * (1 + adj)
        away = model_probs["away_win"] * (1 - adj)
        draw = model_probs["draw"]  # 保持原值，再做归一化

        total = home + draw + away
        if total <= 0:
            return model_probs  # 防御性：异常则返回原值

        adjusted = {
            "home_win": home / total,
            "draw": draw / total,
            "away_win": away / total,
        }
        return adjusted