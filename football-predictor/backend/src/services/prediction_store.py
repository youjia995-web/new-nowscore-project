import os, json, sqlite3, math
from typing import Optional, List, Dict, Tuple
from ..models.match import MatchInput, PredictionResult, TeamSeasonStats, AdminMatch

class PredictionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_table()

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

    def _ensure_table(self):
        conn = self._connect()
        cur = conn.cursor()
        # 预测日志表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                home_team TEXT,
                away_team TEXT,
                pred_outcome TEXT,
                pred_confidence TEXT,
                pred_home REAL,
                pred_draw REAL,
                pred_away REAL,
                market_home REAL,
                market_draw REAL,
                market_away REAL,
                likely_score TEXT,
                calibration_brier REAL,
                calibration_kl REAL,
                analysis_source TEXT,
                result_json TEXT,
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                actual_outcome TEXT
            )
        """)
        # 索引优化（预测日志常用查询）
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_logs_teams ON prediction_logs(home_team, away_team)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_logs_created ON prediction_logs(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prediction_logs_actual ON prediction_logs(actual_outcome)")
        except Exception:
            pass
        # 球队赛季统计（可手动维护历史 xG/xGA/xPTS）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_season_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                season TEXT NOT NULL,
                xg REAL NOT NULL,
                xga REAL NOT NULL,
                xpts REAL NOT NULL,
                notes TEXT,
                UNIQUE(team, season)
            )
        """)
        conn.commit()
        # 手动比赛记录（用于补充数据库历史或校准）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS matches_manual (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                league TEXT,
                season TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_goals INTEGER,
                away_goals INTEGER,
                home_xg REAL,
                away_xg REAL,
                notes TEXT
            )
        """)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_manual_teams ON matches_manual(home_team, away_team)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_manual_season ON matches_manual(season)")
        except Exception:
            pass
        conn.commit()
        # —— 新增：标定参数表（key 唯一，JSON 存储） ——
        cur.execute("""
            CREATE TABLE IF NOT EXISTS calibration_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value_json TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        # —— 新增：球队每日指标导入表 ——
        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_metrics_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                season TEXT NOT NULL,
                date TEXT NOT NULL,
                league TEXT,
                games_played INTEGER,
                points REAL,
                wins INTEGER,
                draws INTEGER,
                losses INTEGER,
                goals_for INTEGER,
                goals_against INTEGER,
                xg REAL,
                npxg REAL,
                xga REAL,
                npxga REAL,
                xpxgd REAL,
                ppda REAL,
                oppda REAL,
                dc REAL,
                odc REAL,
                xpts REAL,
                UNIQUE(team, season, date)
            )
        """)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_team_metrics_daily_team ON team_metrics_daily(team, season, date)")
        except Exception:
            pass
        conn.commit()
        # —— 迁移：若历史库缺少 games_played 列，补充添加 ——
        try:
            cur.execute("PRAGMA table_info(team_metrics_daily)")
            cols = [r[1] for r in cur.fetchall()]
            if "games_played" not in cols:
                cur.execute("ALTER TABLE team_metrics_daily ADD COLUMN games_played INTEGER")
                conn.commit()
        except Exception:
            # 忽略迁移失败（例如列已存在或旧版 SQLite 不支持 IF NOT EXISTS）
            pass
        # —— 新增：队名映射表（中英文及别名归一化） ——
        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                alias_name TEXT NOT NULL,
                league TEXT,
                lang TEXT,
                UNIQUE(alias_name, league)
            )
        """)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_team_aliases_canonical ON team_aliases(canonical_name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_team_aliases_alias ON team_aliases(alias_name)")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def _label_and_conf(self, res: PredictionResult) -> Tuple[str, str, float]:
        ph = res.fused_home_win_prob or res.home_win_prob
        pd = res.fused_draw_prob or res.draw_prob
        pa = res.fused_away_win_prob or res.away_win_prob
        dist = {'home': ph, 'draw': pd, 'away': pa}
        label = max(dist, key=dist.get)
        peak = dist[label]
        conf = '高' if peak >= 0.6 else ('中' if peak >= 0.45 else '低')
        return label, conf, peak

    def save_prediction(self, match: MatchInput, res: PredictionResult) -> int:
        label, conf, peak = self._label_and_conf(res)
        ls = (res.likely_scores[0].score if (res.likely_scores and hasattr(res.likely_scores[0], "score")) else None)
        brier = (res.calibration_metrics or {}).get("brier_model_vs_market")
        kl = (res.calibration_metrics or {}).get("kl_model_vs_market")
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO prediction_logs
             (home_team, away_team, pred_outcome, pred_confidence, pred_home, pred_draw, pred_away,
              market_home, market_draw, market_away, likely_score, calibration_brier, calibration_kl, analysis_source, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match.home_team.name, match.away_team.name, label, conf,
                res.fused_home_win_prob or res.home_win_prob,
                res.fused_draw_prob or res.draw_prob,
                res.fused_away_win_prob or res.away_win_prob,
                res.market_home_win_prob, res.market_draw_prob, res.market_away_win_prob,
                ls, brier, kl, res.analysis_source, json.dumps(res.dict(), ensure_ascii=False)
            )
        )
        conn.commit()
        rid = cur.lastrowid
        conn.close()
        return rid

    def list_predictions(self, limit: int = 50, offset: int = 0, team: Optional[str] = None, has_result: Optional[bool] = None) -> List[Dict]:
        conn = self._connect()
        cur = conn.cursor()
        sql = "SELECT id, created_at, home_team, away_team, pred_outcome, pred_confidence, pred_home, pred_draw, pred_away, market_home, market_draw, market_away, likely_score, calibration_brier, calibration_kl, analysis_source, actual_home_goals, actual_away_goals, actual_outcome FROM prediction_logs"
        params: List = []
        where: List[str] = []
        if team:
            where.append("(home_team LIKE ? OR away_team LIKE ?)"
            )
            like = f"%{team}%"
            params += [like, like]
        if has_result is not None:
            where.append("actual_outcome IS " + ("NOT NULL" if has_result else "NULL"))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        items = []
        for (id, created_at, home_team, away_team, pred_outcome, pred_confidence, ph, pd, pa, mh, md, ma, ls, brier, kl, src, ah, aa, ao) in rows:
            items.append({
                "id": id, "created_at": created_at,
                "home_team": home_team, "away_team": away_team,
                "pred_outcome": pred_outcome, "pred_confidence": pred_confidence,
                "pred_probs": {"home": ph, "draw": pd, "away": pa},
                "market_probs": {"home": mh, "draw": md, "away": ma},
                "likely_score": ls,
                "calibration": {"brier": brier, "kl": kl},
                "analysis_source": src,
                "actual": {"home_goals": ah, "away_goals": aa, "outcome": ao} if ao else None
            })
        return items

    def list_predictions_by_date(self, date: str, team: Optional[str] = None, has_result: Optional[bool] = None) -> List[Dict]:
        conn = self._connect()
        cur = conn.cursor()
        sql = (
            "SELECT id, created_at, home_team, away_team, pred_outcome, pred_confidence, pred_home, pred_draw, pred_away, "
            "market_home, market_draw, market_away, likely_score, calibration_brier, calibration_kl, analysis_source, "
            "actual_home_goals, actual_away_goals, actual_outcome FROM prediction_logs WHERE date(created_at) = ?"
        )
        params: List = [date]
        if team:
            sql += " AND (home_team LIKE ? OR away_team LIKE ?)"
            like = f"%{team}%"
            params += [like, like]
        if has_result is not None:
            sql += " AND actual_outcome IS " + ("NOT NULL" if has_result else "NULL")
        sql += " ORDER BY id DESC"
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        items = []
        for (id, created_at, home_team, away_team, pred_outcome, pred_confidence, ph, pd, pa, mh, md, ma, ls, brier, kl, src, ah, aa, ao) in rows:
            items.append({
                "id": id, "created_at": created_at,
                "home_team": home_team, "away_team": away_team,
                "pred_outcome": pred_outcome, "pred_confidence": pred_confidence,
                "pred_probs": {"home": ph, "draw": pd, "away": pa},
                "market_probs": {"home": mh, "draw": md, "away": ma},
                "likely_score": ls,
                "calibration": {"brier": brier, "kl": kl},
                "analysis_source": src,
                "actual": {"home_goals": ah, "away_goals": aa, "outcome": ao} if ao else None
            })
        return items

    def calendar_counts(self, month: str) -> List[Dict]:
        conn = self._connect()
        cur = conn.cursor()
        sql = "SELECT date(created_at) AS d, COUNT(*) FROM prediction_logs WHERE strftime('%Y-%m', created_at) = ? GROUP BY d ORDER BY d"
        cur.execute(sql, [month])
        rows = cur.fetchall()
        conn.close()
        return [{"date": d, "count": c} for (d, c) in rows]

    def get_prediction(self, id: int) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT result_json FROM prediction_logs WHERE id = ?", (id,))
        row = cur.fetchone()
        conn.close()
        return json.loads(row[0]) if row and row[0] else {}

    def update_actual_result(self, id: int, home_goals: int, away_goals: int) -> Dict:
        ao = 'H' if home_goals > away_goals else ('A' if away_goals > home_goals else 'D')
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("UPDATE prediction_logs SET actual_home_goals=?, actual_away_goals=?, actual_outcome=? WHERE id=?", (home_goals, away_goals, ao, id))
        conn.commit()
        conn.close()
        return {"id": id, "actual_outcome": ao}

    def try_fill_actual_from_matches(self, id: int) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT home_team, away_team, date(created_at) FROM prediction_logs WHERE id = ?", (id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return {"found": False}
        home_team, away_team, d = row
        cur.execute("SELECT home_goals, away_goals FROM matches_manual WHERE home_team=? AND away_team=? AND date=? LIMIT 1", (home_team, away_team, d))
        m = cur.fetchone()
        conn.close()
        if m and (m[0] is not None) and (m[1] is not None):
            return self.update_actual_result(id, int(m[0]), int(m[1]))
        return {"found": False}

    def upsert_team_season_stats(self, stats: TeamSeasonStats) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO team_season_stats (team, season, xg, xga, xpts, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(team, season) DO UPDATE SET
                xg=excluded.xg,
                xga=excluded.xga,
                xpts=excluded.xpts,
                notes=excluded.notes
        """, (stats.team, stats.season, stats.xg, stats.xga, stats.xpts, stats.notes))
        conn.commit()
        # 获取 id
        cur.execute("SELECT id FROM team_season_stats WHERE team=? AND season=?", (stats.team, stats.season))
        row = cur.fetchone()
        conn.close()
        return {"id": row[0] if row else None, "team": stats.team, "season": stats.season}

    def list_team_season_stats(self, team: Optional[str] = None, season: Optional[str] = None, league: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict]:
        conn = self._connect()
        cur = conn.cursor()
        # 检查是否存在联赛列（兼容中文/英文列名）
        league_col = None
        try:
            cur.execute("PRAGMA table_info(team_season_stats)")
            cols = [row[1] for row in cur.fetchall()]
            for cand in ["联赛名称", "league", "联赛", "league_name"]:
                if cand in cols:
                    league_col = cand
                    break
        except Exception:
            league_col = None
        # 构造查询字段；若无联赛列则通过 matches_manual 推断
        base_select = "SELECT id, team, season, xg, xga, xpts, COALESCE(notes, '')"
        if league_col:
            base_select += f", COALESCE({league_col}, '') AS league"
        else:
            computed_league = "(SELECT mm.league FROM matches_manual mm WHERE mm.season = team_season_stats.season AND (mm.home_team = team_season_stats.team OR mm.away_team = team_season_stats.team) ORDER BY mm.id DESC LIMIT 1)"
            base_select += f", COALESCE({computed_league}, '') AS league"
        base_select += " FROM team_season_stats"
        params: List = []
        conditions: List[str] = []
        if team:
            conditions.append("team = ?")
            params.append(team)
        if season:
            conditions.append("season = ?")
            params.append(season)
        if league:
            if league_col:
                conditions.append(f"{league_col} = ?")
                params.append(league)
            else:
                # 通过存在于 matches_manual 的联赛过滤
                conditions.append("EXISTS (SELECT 1 FROM matches_manual mm WHERE mm.season = team_season_stats.season AND mm.league = ? AND (mm.home_team = team_season_stats.team OR mm.away_team = team_season_stats.team))")
                params.append(league)
        if conditions:
            base_select += " WHERE " + " AND ".join(conditions)
        base_select += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(base_select, tuple(params))
        rows = cur.fetchall()
        conn.close()
        items: List[Dict] = []
        for r in rows:
            item = {
                "id": r[0],
                "team": r[1],
                "season": r[2],
                "xg": float(r[3]),
                "xga": float(r[4]),
                "xpts": float(r[5]),
                "notes": r[6],
            }
            if len(r) >= 8:
                item["league"] = r[7]
            items.append(item)
        return items

    def get_team_season_stats(self, team: str, season: Optional[str] = None) -> Optional[Dict]:
        conn = self._connect()
        cur = conn.cursor()
        if season:
            cur.execute(
                "SELECT id, team, season, xg, xga, xpts, COALESCE(notes, '') FROM team_season_stats WHERE team=? AND season=? ORDER BY id DESC LIMIT 1",
                (team, season)
            )
        else:
            cur.execute(
                "SELECT id, team, season, xg, xga, xpts, COALESCE(notes, '') FROM team_season_stats WHERE team=? ORDER BY id DESC LIMIT 1",
                (team,)
            )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row[0],
            "team": row[1],
            "season": row[2],
            "xg": float(row[3]),
            "xga": float(row[4]),
            "xpts": float(row[5]),
            "notes": row[6],
        }

    def insert_match_manual(self, match: AdminMatch) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO matches_manual (date, league, season, home_team, away_team, home_goals, away_goals, home_xg, away_xg, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match.date, match.league, match.season,
            match.home_team, match.away_team,
            match.home_goals, match.away_goals,
            match.home_xg, match.away_xg,
            match.notes
        ))
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return {"id": new_id}

    # —— 新增：写入原始比赛表（字段对齐） ——
    def _table_exists(self, conn: sqlite3.Connection, table: str) -> bool:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (table,))
        return cur.fetchone() is not None

    def insert_match_original(self, match: AdminMatch) -> Dict:
        """
        若存在原库表 `matches_data`，写入与原字段匹配的数据；否则回退到 `matches_manual`。
        字段映射核心：
        - 基础：赛季(`比赛信息`)、联赛(`unnamed_2`)、轮次(`unnamed_3`)、主队(`基本面_2`)、客队(`基本面_9`)
        - 排名：主队排名(`基本面_3`)、客队排名(`基本面_10`)
        - 半/全场：半场主客进球(`半全场数据_2`,`半全场数据_3`)、半场比分(`半全场数据_5`)、全场主客进球(`半全场数据_6`,`半全场数据_7`)、总进球(`半全场数据_8`)、全场赛果(`半全场数据_全场赛果`)
        - 公司赔率（四家）：365 / 皇冠（crown） / 澳门 / 易胜博 → 胜平负初/终盘与让球初/终盘（线、标签、水位）
        """
        conn = self._connect()
        try:
            if self._table_exists(conn, "matches_data"):
                cur = conn.cursor()
                # 计算中文赛果与总进球/半场比分
                result_cn = None
                ft_total = None
                ht_score = None
                if match.home_goals is not None and match.away_goals is not None:
                    result_cn = "胜" if match.home_goals > match.away_goals else ("负" if match.home_goals < match.away_goals else "平")
                    ft_total = (match.home_goals + match.away_goals)
                if match.half_home_goals is not None and match.half_away_goals is not None:
                    ht_score = f"{match.half_home_goals}-{match.half_away_goals}"
                cols: List[str] = []
                vals: List = []
                def add(col: str, val):
                    if val is not None:
                        cols.append(col)
                        vals.append(val)
                # —— 基础信息 ——
                add("比赛信息", match.season)
                add("unnamed_2", match.league)
                add("unnamed_3", match.round)
                add("基本面_2", match.home_team)
                add("基本面_9", match.away_team)
                add("基本面_3", match.home_ranking)
                add("基本面_10", match.away_ranking)
                # —— 半/全场 ——
                add("半全场数据_2", match.half_home_goals)
                add("半全场数据_3", match.half_away_goals)
                add("半全场数据_5", ht_score)
                add("半全场数据_6", match.home_goals)
                add("半全场数据_7", match.away_goals)
                add("半全场数据_8", ft_total)
                add("半全场数据_全场赛果", result_cn)
                # —— 公司赔率（胜平负 + 让球初/终盘） ——
                comp_prefix = {
                    "365": "365",
                    "皇冠（crown）": "皇冠（crown）",
                    "澳门": "澳门",
                    "易胜博": "易胜博",
                }
                def parse_line(text: Optional[str]) -> Optional[float]:
                    try:
                        s = str(text or "").strip()
                        import re
                        # 提取括号或文本中的数字，如 "两球半(2.5)" → 2.5
                        m = re.search(r"([0-9]+(?:\\.[0-9]+)?)", s)
                        return float(m.group(1)) if m else None
                    except Exception:
                        return None
                if getattr(match, "odds_data", None):
                    for od in (match.odds_data or []):
                        prefix = comp_prefix.get(od.company)
                        if not prefix:
                            continue
                        # 胜平负三项（初/终盘）
                        add(f"{prefix}_初赔", od.initial_home_win)
                        add(f"{prefix}", od.initial_draw)
                        add(f"{prefix}_2", od.initial_away_win)
                        add(f"{prefix}_终赔", od.final_home_win)
                        add(f"{prefix}_3", od.final_draw)
                        add(f"{prefix}_4", od.final_away_win)
                        # 让球初/终盘：线、标签、水位（以主/客水位分别映射）
                        add(f"{prefix}_初盘", parse_line(od.initial_handicap))
                        add(f"{prefix}_5", od.initial_handicap)
                        add(f"{prefix}_6", od.initial_handicap_home_odds)
                        add(f"{prefix}_终盘", parse_line(od.final_handicap))
                        add(f"{prefix}_7", od.final_handicap)
                        add(f"{prefix}_8", od.final_handicap_away_odds)
                # 若没有任何可写入列，则回退
                if not cols:
                    return self.insert_match_manual(match)
                sql = f"INSERT INTO matches_data ({', '.join(cols)}) VALUES ({', '.join(['?']*len(cols))})"
                cur.execute(sql, vals)
                conn.commit()
                return {"table": "matches_data", "id": cur.lastrowid, "columns_written": cols}
            else:
                # 回退到手动表
                return self.insert_match_manual(match)
        finally:
            conn.close()

    def backtest_summary(self) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT pred_home, pred_draw, pred_away, actual_outcome FROM prediction_logs WHERE actual_outcome IS NOT NULL")
        rows = cur.fetchall()
        conn.close()
        n = len(rows)
        if n == 0:
            return {"samples": 0, "accuracy": None, "avg_brier": None, "avg_logloss": None, "reliability": [], "ece": None}
        correct = 0
        briers: List[float] = []
        loglosses: List[float] = []
        buckets = [{"sum_prob": 0.0, "count": 0, "correct": 0} for _ in range(20)]
        for ph, pd, pa, ao in rows:
            if (ao == 'H' and ph >= pd and ph >= pa) or (ao == 'D' and pd >= ph and pd >= pa) or (ao == 'A' and pa >= ph and pa >= pd):
                correct += 1
            oh, od, oa = (1.0, 0.0, 0.0) if ao == 'H' else ((0.0, 1.0, 0.0) if ao == 'D' else (0.0, 0.0, 1.0))
            briers.append((ph - oh) ** 2 + (pd - od) ** 2 + (pa - oa) ** 2)
            p_actual = {'H': ph, 'D': pd, 'A': pa}[ao]
            loglosses.append(-math.log(max(p_actual, 1e-9)))
            peak = max(ph, pd, pa)
            pred_label = 'H' if ph >= pd and ph >= pa else ('D' if pd >= ph and pd >= pa else 'A')
            hit = (pred_label == ao)
            idx = min(int(peak * 20), 19)
            buckets[idx]["sum_prob"] += peak
            buckets[idx]["count"] += 1
            buckets[idx]["correct"] += (1 if hit else 0)
        reliability = []
        ece_sum = 0.0
        for i, b in enumerate(buckets):
            lo, hi = i / 20.0, (i + 1) / 20.0
            if b["count"] > 0:
                mean_prob = b["sum_prob"] / b["count"]
                acc = b["correct"] / b["count"]
                reliability.append({"bin": f"{lo:.2f}-{hi:.2f}", "mean_prob": mean_prob, "accuracy": acc, "count": b["count"]})
                ece_sum += abs(acc - mean_prob) * (b["count"] / n)
            else:
                reliability.append({"bin": f"{lo:.2f}-{hi:.2f}", "mean_prob": None, "accuracy": None, "count": 0})
        return {
            "samples": n,
            "accuracy": correct / n,
            "avg_brier": float(sum(briers) / n),
            "avg_logloss": float(sum(loglosses) / n),
            "reliability": reliability,
            "ece": ece_sum,
        }

    def backtest_calibration_by_class(self) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT pred_home, pred_draw, pred_away, actual_outcome FROM prediction_logs WHERE actual_outcome IS NOT NULL")
        rows = cur.fetchall()
        conn.close()
        n = len(rows)
        if n == 0:
            return {
                "samples": 0,
                "per_class_accuracy": {"home": None, "draw": None, "away": None},
                "reliability_by_class": {"home": [], "draw": [], "away": []},
                "ece_by_class": {"home": None, "draw": None, "away": None},
                "ece_macro": None,
            }
        cls_names = ["home", "draw", "away"]
        buckets = {c: [{"sum_prob": 0.0, "count": 0, "positives": 0} for _ in range(20)] for c in cls_names}
        pos_counts = {"home": 0, "draw": 0, "away": 0}
        for ph, pd, pa, ao in rows:
            dist = {"home": ph, "draw": pd, "away": pa}
            actual_map = {"H": "home", "D": "draw", "A": "away"}
            actual_cls = actual_map.get(ao)
            if actual_cls in pos_counts:
                pos_counts[actual_cls] += 1
            for c in cls_names:
                p = dist[c]
                if p is None:
                    continue
                idx = min(int(p * 20), 19)
                buckets[c][idx]["sum_prob"] += p
                buckets[c][idx]["count"] += 1
                if actual_cls == c:
                    buckets[c][idx]["positives"] += 1
        reliability_by_class = {"home": [], "draw": [], "away": []}
        ece_by_class = {}
        for c in cls_names:
            ece_sum = 0.0
            total = n
            for i, b in enumerate(buckets[c]):
                lo, hi = i / 20.0, (i + 1) / 20.0
                if b["count"] > 0:
                    mean_prob = b["sum_prob"] / b["count"]
                    acc = b["positives"] / b["count"]
                    reliability_by_class[c].append({"bin": f"{lo:.2f}-{hi:.2f}", "mean_prob": mean_prob, "accuracy": acc, "count": b["count"]})
                    ece_sum += abs(acc - mean_prob) * (b["count"] / total)
                else:
                    reliability_by_class[c].append({"bin": f"{lo:.2f}-{hi:.2f}", "mean_prob": None, "accuracy": None, "count": 0})
            ece_by_class[c] = ece_sum
        ece_macro = sum(ece_by_class.values()) / len(cls_names)
        per_class_accuracy = {
            "home": pos_counts["home"] / n,
            "draw": pos_counts["draw"] / n,
            "away": pos_counts["away"] / n,
        }
        return {
            "samples": n,
            "per_class_accuracy": per_class_accuracy,
            "reliability_by_class": reliability_by_class,
            "ece_by_class": ece_by_class,
            "ece_macro": ece_macro,
        }

        # ...
        # ...
    def upsert_team_metrics_daily(self, metrics: Dict) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        # 统一列集合
        keys = [
            "team","season","date","league","games_played","points","wins","draws","losses",
            "goals_for","goals_against","xg","npxg","xga","npxga","xpxgd","ppda","oppda","dc","odc","xpts"
        ]
        vals = [metrics.get(k) for k in keys]
        cur.execute(
            f"""
            INSERT INTO team_metrics_daily ({', '.join(keys)})
            VALUES ({', '.join(['?']*len(keys))})
            ON CONFLICT(team, season, date) DO UPDATE SET
                league=excluded.league,
                games_played=excluded.games_played,
                points=excluded.points,
                wins=excluded.wins,
                draws=excluded.draws,
                losses=excluded.losses,
                goals_for=excluded.goals_for,
                goals_against=excluded.goals_against,
                xg=excluded.xg,
                npxg=excluded.npxg,
                xga=excluded.xga,
                npxga=excluded.npxga,
                xpxgd=excluded.xpxgd,
                ppda=excluded.ppda,
                oppda=excluded.oppda,
                dc=excluded.dc,
                odc=excluded.odc,
                xpts=excluded.xpts
            """,
            vals
        )
        conn.commit()
        rid = cur.lastrowid
        conn.close()
        return {"id": rid}

    def delete_team_metrics_daily_by_seasons(self, seasons: List[str]) -> int:
        """删除指定赛季范围内的 team_metrics_daily 旧数据，用于覆盖式导入。"""
        if not seasons:
            return 0
        conn = self._connect()
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(seasons))
        cur.execute(f"DELETE FROM team_metrics_daily WHERE season IN ({placeholders})", seasons)
        affected = cur.rowcount if hasattr(cur, "rowcount") else 0
        conn.commit()
        conn.close()
        return int(affected or 0)

    def get_team_latest_metrics(self, team: str, season: Optional[str] = None) -> Optional[Dict]:
        conn = self._connect()
        cur = conn.cursor()
        if season:
            cur.execute(
                """
                SELECT team, season, date, league, games_played, points, wins, draws, losses, goals_for, goals_against,
                       xg, npxg, xga, npxga, xpxgd, ppda, oppda, dc, odc, xpts
                FROM team_metrics_daily
                WHERE team=? AND season=?
                ORDER BY date DESC, id DESC LIMIT 1
                """,
                (team, season)
            )
        else:
            cur.execute(
                """
                SELECT team, season, date, league, games_played, points, wins, draws, losses, goals_for, goals_against,
                       xg, npxg, xga, npxga, xpxgd, ppda, oppda, dc, odc, xpts
                FROM team_metrics_daily
                WHERE team=?
                ORDER BY date DESC, id DESC LIMIT 1
                """,
                (team,)
            )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        (team, season, date, league, gp_db, points, wins, draws, losses, gf, ga,
         xg, npxg, xga, npxga, xpxgd, ppda, oppda, dc, odc, xpts) = row
        # 规范化已赛场次：将空字符串或不可解析值视为缺失，并用 W+D+L 兜底
        gp = None
        try:
            if gp_db is not None:
                if isinstance(gp_db, str):
                    s = gp_db.strip()
                    gp = int(s) if s != '' else None
                elif isinstance(gp_db, (int, float)):
                    # SQLite 可能返回 float，统一转为 int
                    gp = int(gp_db)
        except Exception:
            gp = None
        try:
            if (gp is None) and (wins is not None or draws is not None or losses is not None):
                gp = int((wins or 0) + (draws or 0) + (losses or 0))
        except Exception:
            pass
        return {
            "team": team, "season": season, "date": date, "league": league,
            "points": points,
            "wins": wins, "draws": draws, "losses": losses,
            "goals_for": gf, "goals_against": ga,
            "xg": xg, "npxg": npxg, "xga": xga, "npxga": npxga, "xpxgd": xpxgd,
            "ppda": ppda, "oppda": oppda, "dc": dc, "odc": odc, "xpts": xpts,
            "games_played": gp
        }

    def resolve_team_name(self, name: Optional[str], league: Optional[str] = None) -> str:
        """根据映射表解析球队规范名；若未命中，按常见中文别名归一并返回清洗值。"""
        if not name:
            return ""
        s = str(name).strip()
        try:
            import re
            s = re.sub(r"\[.*?\]", "", s)
            s = re.sub(r"\(.*?\)", "", s)
        except Exception:
            pass
        s_key = s.replace(" ", "")
        conn = self._connect()
        cur = conn.cursor()
        try:
            row = None
            if league:
                cur.execute(
                    "SELECT canonical_name FROM team_aliases WHERE alias_name=? AND league=? ORDER BY id DESC LIMIT 1",
                    (s_key, league)
                )
                row = cur.fetchone()
                if not row:
                    # 回退：当映射中没有联赛时也能匹配
                    cur.execute(
                        "SELECT canonical_name FROM team_aliases WHERE alias_name=? AND (league IS NULL OR league='') ORDER BY id DESC LIMIT 1",
                        (s_key,)
                    )
                    row = cur.fetchone()
            else:
                cur.execute("SELECT canonical_name FROM team_aliases WHERE alias_name=? ORDER BY id DESC LIMIT 1", (s_key,))
                row = cur.fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            pass
        finally:
            try:
                conn.close()
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
        return aliases.get(s_key, s_key)

    def upsert_team_alias(self, canonical_name: str, alias_name: str, league: Optional[str] = None, lang: Optional[str] = None) -> Dict:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO team_aliases (canonical_name, alias_name, league, lang)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(alias_name, league) DO UPDATE SET
                canonical_name=excluded.canonical_name,
                lang=excluded.lang
            """,
            (canonical_name, alias_name, league, lang)
        )
        conn.commit()
        # 获取 id
        cur.execute("SELECT id FROM team_aliases WHERE alias_name=? AND (league IS ? OR league=?) ORDER BY id DESC LIMIT 1", (alias_name, league, league))
        row = cur.fetchone()
        conn.close()
        return {"id": row[0] if row else None, "alias_name": alias_name, "canonical_name": canonical_name}

    def bulk_upsert_team_aliases(self, items: List[Dict]) -> Dict:
        """批量写入队名映射。items: {canonical_name, alias_name, league?, lang?}"""
        if not items:
            return {"imported": 0}
        conn = self._connect()
        cur = conn.cursor()
        cnt = 0
        for it in items:
            try:
                canonical_name = str(it.get("canonical_name") or "").strip()
                alias_name = str(it.get("alias_name") or "").strip()
                league = it.get("league")
                lang = it.get("lang")
                if not canonical_name or not alias_name:
                    continue
                cur.execute(
                    """
                    INSERT INTO team_aliases (canonical_name, alias_name, league, lang)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(alias_name, league) DO UPDATE SET
                        canonical_name=excluded.canonical_name,
                        lang=excluded.lang
                    """,
                    (canonical_name, alias_name.replace(" ", ""), league, lang)
                )
                cnt += 1
            except Exception:
                pass
        conn.commit()
        conn.close()
        return {"imported": cnt}
