import os, sqlite3
from typing import Optional, List, Dict, Any

class RosterStore:
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS team_roster (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT NOT NULL,
                season TEXT NOT NULL,
                player_name TEXT NOT NULL,
                position TEXT,
                jersey_number TEXT,
                nationality TEXT,
                age INTEGER,
                height_cm REAL,
                weight_kg REAL,
                preferred_foot TEXT,
                appearances INTEGER,
                starts INTEGER,
                minutes INTEGER,
                goals INTEGER,
                assists INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                injured INTEGER DEFAULT 0,
                suspended INTEGER DEFAULT 0,
                market_value REAL,
                rating REAL,
                notes TEXT,
                UNIQUE(team, season, player_name)
            );
            """
        )
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_team_roster_team_season ON team_roster(team, season)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_team_roster_player ON team_roster(player_name)")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def upsert_row(self, row: Dict[str, Any]) -> None:
        required = ["team", "season", "player_name"]
        for r in required:
            if not row.get(r):
                raise ValueError(f"Missing required field: {r}")
        # 规范布尔到 0/1
        for k in ["injured", "suspended"]:
            v = row.get(k)
            if isinstance(v, str):
                lv = v.strip().lower()
                row[k] = 1 if lv in {"1","true","yes","y","是","有","伤病","停赛"} else 0
            else:
                row[k] = int(bool(v)) if v is not None else 0
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO team_roster (
                team, season, player_name, position, jersey_number, nationality,
                age, height_cm, weight_kg, preferred_foot, appearances, starts,
                minutes, goals, assists, yellow_cards, red_cards, injured, suspended,
                market_value, rating, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(team, season, player_name) DO UPDATE SET
                position=excluded.position,
                jersey_number=excluded.jersey_number,
                nationality=excluded.nationality,
                age=excluded.age,
                height_cm=excluded.height_cm,
                weight_kg=excluded.weight_kg,
                preferred_foot=excluded.preferred_foot,
                appearances=excluded.appearances,
                starts=excluded.starts,
                minutes=excluded.minutes,
                goals=excluded.goals,
                assists=excluded.assists,
                yellow_cards=excluded.yellow_cards,
                red_cards=excluded.red_cards,
                injured=excluded.injured,
                suspended=excluded.suspended,
                market_value=excluded.market_value,
                rating=excluded.rating,
                notes=excluded.notes
            """,
            [
                row.get("team"), row.get("season"), row.get("player_name"), row.get("position"), row.get("jersey_number"), row.get("nationality"),
                row.get("age"), row.get("height_cm"), row.get("weight_kg"), row.get("preferred_foot"), row.get("appearances"), row.get("starts"),
                row.get("minutes"), row.get("goals"), row.get("assists"), row.get("yellow_cards"), row.get("red_cards"), row.get("injured"), row.get("suspended"),
                row.get("market_value"), row.get("rating"), row.get("notes")
            ]
        )
        conn.commit()
        conn.close()

    def bulk_import(self, df, team: Optional[str] = None, season: Optional[str] = None) -> Dict[str, Any]:
        # 统一列名小写
        df.columns = [str(c).strip().lower() for c in df.columns]
        alias = {
            # 基本身份
            "name": "player_name",
            "player": "player_name",
            "球员": "player_name",
            "球员姓名": "player_name",
            "姓名": "player_name",
            "队员": "player_name",
            "位置": "position",
            "pos": "position",
            "号码": "jersey_number",
            "球衣号码": "jersey_number",
            "number": "jersey_number",
            "国籍": "nationality",
            "age": "age",
            "年龄": "age",
            "身高": "height_cm",
            "身高cm": "height_cm",
            "体重": "weight_kg",
            "惯用脚": "preferred_foot",
            # 贡献指标
            "apps": "appearances",
            "出场": "appearances",
            "出场数": "appearances",
            "starts": "starts",
            "首发": "starts",
            "分钟": "minutes",
            "出场分钟": "minutes",
            "goals": "goals",
            "进球": "goals",
            "assists": "assists",
            "助攻": "assists",
            "黄牌": "yellow_cards",
            "红牌": "red_cards",
            # 伤停与状态
            "injury": "injured",
            "injured": "injured",
            "伤病": "injured",
            "受伤": "injured",
            "suspension": "suspended",
            "suspended": "suspended",
            "停赛": "suspended",
            # 评分与身价
            "rating": "rating",
            "评分": "rating",
            "market_value": "market_value",
            "身价": "market_value",
            # 团队字段
            "team": "team",
            "球队": "team",
            "season": "season",
            "赛季": "season",
            "备注": "notes",
        }
        # 列别名映射
        for k, v in alias.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        inserted, updated, skipped = 0, 0, 0
        conn = self._connect()
        cur = conn.cursor()
        for _, r in df.iterrows():
            row = {c: (None if (pd.isna(r[c]) if 'pd' in globals() else (r[c] is None)) else r[c]) for c in df.columns}
            # 默认 team/season 填充
            row_team = str(row.get("team") or team or "").strip()
            row_season = str(row.get("season") or season or "").strip()
            player_name = str(row.get("player_name") or "").strip()
            if not row_team or not row_season or not player_name:
                skipped += 1
                continue
            # 规范布尔
            for k in ["injured", "suspended"]:
                v = row.get(k)
                if isinstance(v, str):
                    lv = v.strip().lower()
                    row[k] = 1 if lv in {"1","true","yes","y","是","有","伤病","停赛"} else 0
                else:
                    row[k] = int(bool(v)) if v is not None else 0
            try:
                cur.execute(
                    """
                    INSERT INTO team_roster (
                        team, season, player_name, position, jersey_number, nationality,
                        age, height_cm, weight_kg, preferred_foot, appearances, starts,
                        minutes, goals, assists, yellow_cards, red_cards, injured, suspended,
                        market_value, rating, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(team, season, player_name) DO UPDATE SET
                        position=excluded.position,
                        jersey_number=excluded.jersey_number,
                        nationality=excluded.nationality,
                        age=excluded.age,
                        height_cm=excluded.height_cm,
                        weight_kg=excluded.weight_kg,
                        preferred_foot=excluded.preferred_foot,
                        appearances=excluded.appearances,
                        starts=excluded.starts,
                        minutes=excluded.minutes,
                        goals=excluded.goals,
                        assists=excluded.assists,
                        yellow_cards=excluded.yellow_cards,
                        red_cards=excluded.red_cards,
                        injured=excluded.injured,
                        suspended=excluded.suspended,
                        market_value=excluded.market_value,
                        rating=excluded.rating,
                        notes=excluded.notes
                    """,
                    [
                        row_team, row_season, player_name, row.get("position"), row.get("jersey_number"), row.get("nationality"),
                        row.get("age"), row.get("height_cm"), row.get("weight_kg"), row.get("preferred_foot"), row.get("appearances"), row.get("starts"),
                        row.get("minutes"), row.get("goals"), row.get("assists"), row.get("yellow_cards"), row.get("red_cards"), row.get("injured"), row.get("suspended"),
                        row.get("market_value"), row.get("rating"), row.get("notes")
                    ]
                )
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
            except Exception:
                skipped += 1
        conn.commit()
        conn.close()
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def list_roster(self, team: str, season: str) -> List[Dict[str, Any]]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT player_name, position, jersey_number, nationality, age, appearances, starts, minutes, goals, assists, yellow_cards, red_cards, injured, suspended, market_value, rating, notes FROM team_roster WHERE team=? AND season=? ORDER BY appearances DESC, goals DESC",
            [team, season]
        )
        rows = cur.fetchall()
        conn.close()
        items = []
        for (name, pos, num, nat, age, apps, starts, mins, goals, assists, yc, rc, injured, suspended, mv, rating, notes) in rows:
            items.append({
                "player_name": name,
                "position": pos,
                "jersey_number": num,
                "nationality": nat,
                "age": age,
                "appearances": apps,
                "starts": starts,
                "minutes": mins,
                "goals": goals,
                "assists": assists,
                "yellow_cards": yc,
                "red_cards": rc,
                "injured": injured,
                "suspended": suspended,
                "market_value": mv,
                "rating": rating,
                "notes": notes,
            })
        return items

    def availability_index(self, team: str, season: str) -> Optional[float]:
        """计算阵容可用性指数：基于贡献加权，顶级核心（最多10人）可用占比。
        返回 [0,1]，None 表示数据不足。
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT player_name, appearances, starts, minutes, goals, assists, rating, market_value, injured, suspended FROM team_roster WHERE team=? AND season=?",
            [team, season]
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return None
        def contrib(r):
            (_, apps, starts, mins, goals, assists, rating, mv, injured, suspended) = r
            # 贡献分：兼顾进攻、出场与评分/身价（若可用）
            g = float(goals or 0)
            a = float(assists or 0)
            s = float(starts or 0)
            ap = float(apps or 0)
            m = float(mins or 0)
            rt = float(rating or 0)
            v = float(mv or 0)
            base = 0.6 * g + 0.4 * a + 0.05 * s + 0.02 * ap + 0.02 * (m / 90.0)
            # 评分与身价微调（限幅）
            base += min(0.5, max(0.0, rt * 0.1))
            base += min(0.3, max(0.0, v / 100.0))
            return base
        scored = sorted(rows, key=contrib, reverse=True)[:10]
        total = sum(contrib(r) for r in scored)
        if total <= 0:
            return None
        available = sum(contrib(r) for r in scored if int(r[-2] or 0) == 0 and int(r[-1] or 0) == 0)
        return max(0.0, min(1.0, available / total))

# pandas 仅在 bulk_import 中按需使用，延迟导入以减少后端启动时依赖
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None