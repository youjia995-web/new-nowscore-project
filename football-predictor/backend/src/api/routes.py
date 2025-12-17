from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional, Dict
from ..models.match import MatchInput, PredictionResult, TeamSeasonStats, AdminMatch
# 惰性引入耗时/重依赖模块，避免服务因缺少 SciPy/Pandas 无法启动
try:
    from ..engines.poisson_engine import PoissonEngine  # 依赖 scipy
except Exception:
    PoissonEngine = None
try:
    from ..engines.bayes_poisson_engine import BayesPoissonEngine
except Exception:
    BayesPoissonEngine = None
try:
    from ..engines.market_engine import MarketEngine
except Exception:
    MarketEngine = None
try:
    from ..engines.ai_engine import AIEngine  # 依赖 requests
except Exception:
    AIEngine = None
try:
    from ..services.prob_calibration import ProbCalibrationService
except Exception:
    ProbCalibrationService = None
from ..config import settings
from ..services.prediction_store import PredictionStore
from ..services.roster_store import RosterStore
from ..services.formation_store import FormationStore
from ..services.exchange_store import load_exchange_features_from_bytes, load_exchange_features_from_path
import io
# pandas 在使用模板/导入功能时再按需导入

router = APIRouter()
try:
    import torch
    from torch import nn
    from pathlib import Path
    import math
    class TwinGRU(nn.Module):
        def __init__(self, input_dim=32, hidden=64, dropout=0.4, num_classes=3):
            super().__init__()
            self.h_gru = nn.GRU(input_dim, hidden, batch_first=True)
            self.a_gru = nn.GRU(input_dim, hidden, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))
        def forward(self, h_seq, a_seq):
            _, h = self.h_gru(h_seq)
            _, a = self.a_gru(a_seq)
            x = torch.cat([h[-1], a[-1]], dim=1)
            return self.fc(x)
    _GRU_MODEL = None
    _GRU_PATH = str(Path(__file__).resolve().parents[2] / "lstm_gru.pth")
    def _load_gru():
        global _GRU_MODEL
        if _GRU_MODEL is None:
            m = TwinGRU(input_dim=32, hidden=64, dropout=0.4)
            try:
                m.load_state_dict(torch.load(_GRU_PATH, map_location="cpu"))
                m.eval()
                _GRU_MODEL = m
            except Exception:
                _GRU_MODEL = None
    def _parse_handicap(text):
        try:
            s = str(text or "").strip()
            import re
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
            return float(m.group(1)) if m else 0.0
        except Exception:
            return 0.0
    _FEATS = [
        "gf","ga","xgf","xga","is_home","rank","round",
        "odds_h_o","odds_d_o","odds_a_o",
        "odds_h_c","odds_d_c","odds_a_c",
        "ah_line_o","ah_hp_o","ah_ap_o",
        "ah_line_c",
        "prob_h_o","prob_d_o","prob_a_o",
        "prob_h_c","prob_d_c","prob_a_c",
        "delta_h","delta_d","delta_a",
        "over_o","over_c",
        "log_ratio_o","log_ratio_c",
        "ah_p_h_o","ah_p_h_c"
    ]
    def _rec_from(team, is_home, odds):
        gf = float(getattr(team, "goals_for", 0) or 0.0)
        ga = float(getattr(team, "goals_against", 0) or 0.0)
        xgf = float(getattr(team, "xg", 0.0) or 0.0)
        xga = float(getattr(team, "xga", 0.0) or 0.0)
        rank = float(getattr(team, "ranking", 0) or 0.0)
        rnd = float(getattr(team, "games_played", 0) or 0.0)
        oh_o = float(getattr(odds, "initial_home_win", 0.0) or 0.0)
        od_o = float(getattr(odds, "initial_draw", 0.0) or 0.0)
        oa_o = float(getattr(odds, "initial_away_win", 0.0) or 0.0)
        oh_c = float(getattr(odds, "final_home_win", 0.0) or 0.0)
        od_c = float(getattr(odds, "final_draw", 0.0) or 0.0)
        oa_c = float(getattr(odds, "final_away_win", 0.0) or 0.0)
        ah_o = float(_parse_handicap(getattr(odds, "initial_handicap", None)))
        ah_c = float(_parse_handicap(getattr(odds, "final_handicap", None)))
        ah_hp_o = float(getattr(odds, "initial_handicap_home_odds", 0.0) or 0.0)
        ah_ap_o = float(getattr(odds, "initial_handicap_away_odds", 0.0) or 0.0)
        rec = {
            "gf": gf, "ga": ga, "xgf": xgf, "xga": xga,
            "is_home": (1.0 if is_home else 0.0), "rank": rank, "round": rnd,
            "odds_h_o": oh_o, "odds_d_o": od_o, "odds_a_o": oa_o,
            "odds_h_c": oh_c, "odds_d_c": od_c, "odds_a_c": oa_c,
            "ah_line_o": ah_o, "ah_hp_o": ah_hp_o, "ah_ap_o": ah_ap_o,
            "ah_line_c": ah_c
        }
        p_h_o = (1.0 / rec["odds_h_o"]) if rec["odds_h_o"] > 0 else 0.0
        p_d_o = (1.0 / rec["odds_d_o"]) if rec["odds_d_o"] > 0 else 0.0
        p_a_o = (1.0 / rec["odds_a_o"]) if rec["odds_a_o"] > 0 else 0.0
        s_o = p_h_o + p_d_o + p_a_o
        p_h_o_n = (p_h_o / s_o) if s_o > 0 else 0.0
        p_d_o_n = (p_d_o / s_o) if s_o > 0 else 0.0
        p_a_o_n = (p_a_o / s_o) if s_o > 0 else 0.0
        over_o = (p_h_o + p_d_o + p_a_o) - 1.0
        p_h_c = (1.0 / rec["odds_h_c"]) if rec["odds_h_c"] > 0 else 0.0
        p_d_c = (1.0 / rec["odds_d_c"]) if rec["odds_d_c"] > 0 else 0.0
        p_a_c = (1.0 / rec["odds_a_c"]) if rec["odds_a_c"] > 0 else 0.0
        s_c = p_h_c + p_d_c + p_a_c
        p_h_c_n = (p_h_c / s_c) if s_c > 0 else 0.0
        p_d_c_n = (p_d_c / s_c) if s_c > 0 else 0.0
        p_a_c_n = (p_a_c / s_c) if s_c > 0 else 0.0
        over_c = (p_h_c + p_d_c + p_a_c) - 1.0
        delta_h = rec["odds_h_c"] - rec["odds_h_o"]
        delta_d = rec["odds_d_c"] - rec["odds_d_o"]
        delta_a = rec["odds_a_c"] - rec["odds_a_o"]
        log_ratio_o = math.log(rec["odds_a_o"] / rec["odds_h_o"]) if rec["odds_a_o"] > 0 and rec["odds_h_o"] > 0 else 0.0
        log_ratio_c = math.log(rec["odds_a_c"] / rec["odds_h_c"]) if rec["odds_a_c"] > 0 and rec["odds_h_c"] > 0 else 0.0
        rec.update({
            "prob_h_o": p_h_o_n, "prob_d_o": p_d_o_n, "prob_a_o": p_a_o_n,
            "prob_h_c": p_h_c_n, "prob_d_c": p_d_c_n, "prob_a_c": p_a_c_n,
            "delta_h": delta_h, "delta_d": delta_d, "delta_a": delta_a,
            "over_o": over_o, "over_c": over_c,
            "log_ratio_o": log_ratio_o, "log_ratio_c": log_ratio_c,
            "ah_p_h_o": rec["ah_line_o"] * p_h_o_n,
            "ah_p_h_c": rec["ah_line_c"] * p_h_c_n
        })
        return [rec[f] for f in _FEATS]
    def _predict_with_gru(match_data, seq_len=12):
        if _GRU_MODEL is None or not getattr(match_data, "odds_data", None):
            return None
        od = match_data.odds_data[0]
        h_seq = [_rec_from(match_data.home_team, True, od) for _ in range(seq_len)]
        a_seq = [_rec_from(match_data.away_team, False, od) for _ in range(seq_len)]
        hs = torch.tensor([h_seq], dtype=torch.float32)
        as_ = torch.tensor([a_seq], dtype=torch.float32)
        with torch.no_grad():
            logits = _GRU_MODEL(hs, as_)
            prob = torch.softmax(logits, dim=1)[0].tolist()
        return {"home_win": float(prob[0]), "draw": float(prob[1]), "away_win": float(prob[2])}
except Exception:
    _GRU_MODEL = None

# 简易内存缓存：按比赛键（主队|客队）存储交易所快照特征
EXCHANGE_CACHE: Dict[str, Dict] = {}

@router.get("/mock-data")
async def get_mock_data():
    return [
        {
            "home_team": {
                "name": "曼城",
                "ranking": 1,
                "games_played": 20,
                "xg": 45.5,
                "xga": 15.2,
                "xpts": 48.5,
                "goals_for": 68,
                "goals_against": 20,
                "npxg": 43.2,
                "npxga": 14.4,
                "ppda": 7.2,
                "oppda": 10.8,
                "dc": 14,
                "odc": 5
            },
            "away_team": {
                "name": "谢菲联",
                "ranking": 20,
                "games_played": 20,
                "xg": 18.3,
                "xga": 38.6,
                "xpts": 15.8,
                "goals_for": 28,
                "goals_against": 62,
                "npxg": 17.4,
                "npxga": 36.7,
                "ppda": 12.9,
                "oppda": 7.6,
                "dc": 6,
                "odc": 15
            },
            "odds_data": [
                {
                    "company": "威廉希尔",
                    "initial_home_win": 1.15,
                    "initial_draw": 7.50,
                    "initial_away_win": 19.00,
                    "final_home_win": 1.12,
                    "final_draw": 8.00,
                    "final_away_win": 21.00,
                    "initial_handicap": "两球半(2.5)",
                    "final_handicap": "两球半/三球(2.5/3)",
                    "initial_handicap_home_odds": 1.85,
                    "initial_handicap_away_odds": 1.95,
                    "final_handicap_home_odds": 1.80,
                    "final_handicap_away_odds": 2.00
                },
                {
                    "company": "立博",
                    "initial_home_win": 1.16,
                    "initial_draw": 7.25,
                    "initial_away_win": 18.50,
                    "final_home_win": 1.13,
                    "final_draw": 7.75,
                    "final_away_win": 20.00,
                    "initial_handicap": "两球半(2.5)",
                    "final_handicap": "两球半/三球(2.5/3)",
                    "initial_handicap_home_odds": 1.83,
                    "initial_handicap_away_odds": 1.97,
                    "final_handicap_home_odds": 1.78,
                    "final_handicap_away_odds": 2.02
                }
            ]
        },
        {
            "home_team": {
                "name": "利物浦",
                "ranking": 2,
                "games_played": 22,
                "xg": 50.1,
                "xga": 20.3,
                "xpts": 52.7,
                "goals_for": 71,
                "goals_against": 24,
                "npxg": 47.6,
                "npxga": 19.3,
                "ppda": 8.0,
                "oppda": 11.0,
                "dc": 13,
                "odc": 7
            },
            "away_team": {
                "name": "切尔西",
                "ranking": 10,
                "games_played": 22,
                "xg": 35.8,
                "xga": 30.1,
                "xpts": 33.4,
                "goals_for": 55,
                "goals_against": 44,
                "npxg": 34.0,
                "npxga": 28.6,
                "ppda": 11.5,
                "oppda": 9.0,
                "dc": 9,
                "odc": 10
            },
            "odds_data": [
                {
                    "company": "Bet365",
                    "initial_home_win": 1.65,
                    "initial_draw": 4.00,
                    "initial_away_win": 5.00,
                    "final_home_win": 1.60,
                    "final_draw": 4.20,
                    "final_away_win": 5.50,
                    "initial_handicap": "半球/一球(0.5/1)",
                    "final_handicap": "一球(1)",
                    "initial_handicap_home_odds": 1.90,
                    "initial_handicap_away_odds": 1.90,
                    "final_handicap_home_odds": 1.85,
                    "final_handicap_away_odds": 1.95
                }
            ]
        },
        {
            "home_team": {
                "name": "皇家马德里",
                "games_played": 25,
                "xg": 55.2,
                "xga": 18.9,
                "xpts": 60.1,
                "goals_for": 66,
                "goals_against": 27,
                "npxg": 52.5,
                "npxga": 18.0,
                "ppda": 7.8,
                "oppda": 10.2,
                "dc": 15,
                "odc": 4
            },
            "away_team": {
                "name": "巴塞罗那",
                "games_played": 25,
                "xg": 52.8,
                "xga": 22.5,
                "xpts": 58.3,
                "goals_for": 49,
                "goals_against": 41,
                "npxg": 50.0,
                "npxga": 21.3,
                "ppda": 9.0,
                "oppda": 9.8,
                "dc": 12,
                "odc": 6
            },
            "odds_data": [
                {
                    "company": "澳门彩票",
                    "initial_home_win": 2.20,
                    "initial_draw": 3.50,
                    "initial_away_win": 3.20,
                    "final_home_win": 2.10,
                    "final_draw": 3.60,
                    "final_away_win": 3.40,
                    "initial_handicap": "平手/半球(0/0.5)",
                    "final_handicap": "平手/半球(0/0.5)",
                    "initial_handicap_home_odds": 1.95,
                    "initial_handicap_away_odds": 1.85,
                    "final_handicap_home_odds": 1.90,
                    "final_handicap_away_odds": 1.90
                }
            ]
        }
    ]

@router.post("/predict", response_model=PredictionResult)
async def predict_match(match_data: MatchInput, request: Request):
    poisson_engine = PoissonEngine() if PoissonEngine else None
    if poisson_engine is None:
        try:
            from ..engines.poisson_engine import PoissonEngine as _PE
            poisson_engine = _PE()
        except Exception:
            poisson_engine = None
    market_engine = MarketEngine() if MarketEngine else None
    try:
        ai_engine = AIEngine() if AIEngine else None
        if ai_engine:
            ai_engine.api_key = getattr(settings, "deepseek_api_key", "")
            ai_engine.api_base = getattr(settings, "deepseek_api_base", "")
            ai_engine.enabled = bool(ai_engine.api_key)
    except Exception:
        ai_engine = None
    if poisson_engine is None:
        raise HTTPException(status_code=503, detail="预测引擎未就绪：请安装 scipy 以启用 PoissonEngine")
    if settings.require_ai_api and ai_engine and not getattr(ai_engine, 'enabled', False):
        raise HTTPException(status_code=503, detail="AI API required but not configured (set DEEPSEEK_API_KEY)")
    store = PredictionStore(settings.db_path)
    home_raw_name = match_data.home_team.name
    away_raw_name = match_data.away_team.name
    try:
        home_name = store.resolve_team_name(match_data.home_team.name, getattr(match_data, 'league', None))
        away_name = store.resolve_team_name(match_data.away_team.name, getattr(match_data, 'league', None))
        match_data.home_team = match_data.home_team.copy(update={"name": home_name})
        match_data.away_team = match_data.away_team.copy(update={"name": away_name})
    except Exception:
        pass

    # 历史融合权重函数
    def _history_weight(gp: int) -> float:
        gp = int(gp or 0)
        w_max = getattr(settings, "history_blend_max", 0.50)
        mode = getattr(settings, "history_blend_mode", "linear")
        th = getattr(settings, "history_blend_gp_threshold", 12)
        cutoff = getattr(settings, "history_blend_cutoff_gp", 0)
        if cutoff and gp >= int(cutoff):
            return 0.0
        if mode == "exp":
            import math
            hl = max(1, int(getattr(settings, "history_blend_half_life", 6)))
            w = float(w_max) * math.exp(-math.log(2) * gp / hl)
            return float(max(0.0, min(w, w_max)))
        # 默认线性模式
        if gp >= th:
            return 0.0
        return float(w_max) * (max(0, th - gp) / th)

    season_pref = getattr(settings, "history_default_season", None)
    home_hist = None
    away_hist = None
    if getattr(settings, "history_blend_enabled", True):
        try:
            home_hist = store.get_team_season_stats(match_data.home_team.name, season_pref)
            away_hist = store.get_team_season_stats(match_data.away_team.name, season_pref)
        except Exception:
            home_hist, away_hist = None, None

    home_w = _history_weight(match_data.home_team.games_played)
    away_w = _history_weight(match_data.away_team.games_played)
    home_team_adj = match_data.home_team
    away_team_adj = match_data.away_team
    parts = []
    mode_label = "指数衰减" if getattr(settings, "history_blend_mode", "linear") == "exp" else "线性"
    if home_hist:
        home_team_adj = match_data.home_team.copy(update={
            "xg": (1 - home_w) * match_data.home_team.xg + home_w * float(home_hist["xg"]),
            "xga": (1 - home_w) * match_data.home_team.xga + home_w * float(home_hist["xga"]),
            "xpts": (1 - home_w) * match_data.home_team.xpts + home_w * float(home_hist["xpts"]),
        })
        parts.append(f"主队·{home_hist['team']} 使用{home_hist['season']}历史融合（{mode_label}），权重={home_w:.2f}（xG={home_hist['xg']:.2f}, xGA={home_hist['xga']:.2f}, xPTS={home_hist['xpts']:.2f}）")
    if away_hist:
        away_team_adj = match_data.away_team.copy(update={
            "xg": (1 - away_w) * match_data.away_team.xg + away_w * float(away_hist["xg"]),
            "xga": (1 - away_w) * match_data.away_team.xga + away_w * float(away_hist["xga"]),
            "xpts": (1 - away_w) * match_data.away_team.xpts + away_w * float(away_hist["xpts"]),
        })
        parts.append(f"客队·{away_hist['team']} 使用{away_hist['season']}历史融合（{mode_label}），权重={away_w:.2f}（xG={away_hist['xg']:.2f}, xGA={away_hist['xga']:.2f}, xPTS={away_hist['xpts']:.2f}）")
    history_summary = "；".join(parts) if parts else ""

    # —— 阵容可用性融合（可选）：支持临时阵容或赛季库 ——
    try:
        roster = RosterStore(settings.roster_db_path)
        # 赛季优先使用历史融合返回的赛季；否则用默认配置
        home_season = (home_hist or {}).get("season") or season_pref
        away_season = (away_hist or {}).get("season") or season_pref
        roster_parts = []

        def _availability_index_from_items(items):
            try:
                if not items:
                    return None
                def contrib(it):
                    g = float((it.get("goals") or 0) or 0)
                    a = float((it.get("assists") or 0) or 0)
                    s = float((it.get("starts") or 0) or 0)
                    ap = float((it.get("appearances") or 0) or 0)
                    m = float((it.get("minutes") or 0) or 0)
                    rt = float((it.get("rating") or 0) or 0)
                    v = float((it.get("market_value") or 0) or 0)
                    base = 0.6 * g + 0.4 * a + 0.05 * s + 0.02 * ap + 0.02 * (m / 90.0)
                    base += min(0.5, max(0.0, rt * 0.1))
                    base += min(0.3, max(0.0, v / 100.0))
                    return base
                scored = sorted(items, key=contrib, reverse=True)[:10]
                total = sum(contrib(it) for it in scored)
                if total <= 0:
                    return None
                def _flag(v):
                    # 兼容 '1'/'0'、true/false、'是'/'否'
                    if v is None:
                        return 0
                    if isinstance(v, bool):
                        return 1 if v else 0
                    try:
                        s = str(v).strip().lower()
                        if s in ("1", "true", "是", "有"):
                            return 1
                        return 0
                    except Exception:
                        return 0
                available = sum(contrib(it) for it in scored if _flag(it.get("injured")) == 0 and _flag(it.get("suspended")) == 0)
                return max(0.0, min(1.0, available / total))
            except Exception:
                return None

        def _apply_roster_adjust(team_adj, team_name, team_season, label_prefix, items=None):
            ai = None
            label = label_prefix
            if items and isinstance(items, list) and len(items) > 0:
                ai = _availability_index_from_items(items)
                label += "临时阵容可用性="
            else:
                if not team_season:
                    return team_adj
                ai = roster.availability_index(team_name, team_season)
                label += "阵容可用性="
            if ai is None:
                return team_adj
            loss = 1.0 - float(ai)
            # 进攻下调不超过 -30%，防守上调不超过 +30%
            offense_scale = max(0.70, 1.0 - 0.50 * loss)
            defense_scale = min(1.30, 1.0 + 0.30 * loss)
            roster_parts.append(f"{label}{ai:.2f} → 进攻×{offense_scale:.2f}，防守×{defense_scale:.2f}")
            return team_adj.copy(update={
                "xg": float(team_adj.xg) * offense_scale,
                "xga": float(team_adj.xga) * defense_scale,
            })

        home_team_adj = _apply_roster_adjust(
            home_team_adj, match_data.home_team.name, home_season, "主队·", getattr(match_data, "home_roster", None)
        )
        away_team_adj = _apply_roster_adjust(
            away_team_adj, match_data.away_team.name, away_season, "客队·", getattr(match_data, "away_roster", None)
        )
        if roster_parts:
            history_summary = (history_summary + ("；" if history_summary else "") + "阵容可用性调整：" + "；".join(roster_parts)).strip()
    except Exception:
        pass

    # —— 阵型融合：根据 formation.xlsx 的攻防速率对赛季 xG/xGA 进行乘子调整 ——
    try:
        fstore = FormationStore(settings.formation_xlsx_path)
        fparts = []

        def _apply_formation_adjust(team_adj, label_prefix: str, raw_name: Optional[str] = None):
            team_name = team_adj.name
            form = getattr(team_adj, "formation", None)
            metrics = fstore.get_team_formation(team_name, form, season_pref)
            # 回退：若规范中文名未命中，尝试原始英文名
            if not metrics and raw_name:
                try:
                    metrics = fstore.get_team_formation(raw_name, form, season_pref)
                except Exception:
                    metrics = None
            if not metrics:
                return team_adj
            gp = max(team_adj.games_played, 1)
            base_xg_rate = float(team_adj.npxg if isinstance(team_adj.npxg, (int, float)) else team_adj.xg) / gp
            base_xga_rate = float(team_adj.npxga if isinstance(team_adj.npxga, (int, float)) else team_adj.xga) / gp
            xg90 = metrics.get("xg90")
            xga90 = metrics.get("xga90")
            def _safe(val, default):
                try:
                    return float(val)
                except Exception:
                    return default
            xg90 = _safe(xg90, base_xg_rate)
            xga90 = _safe(xga90, base_xga_rate)
            # 乘子（保护上下限±20%）
            def _clamp_mult(m):
                return max(0.80, min(1.20, float(m)))
            offense_mult = _clamp_mult((xg90 / max(base_xg_rate, 1e-6)) if base_xg_rate > 0 else 1.0)
            defense_mult = _clamp_mult((xga90 / max(base_xga_rate, 1e-6)) if base_xga_rate > 0 else 1.0)
            # 对总量进行等比缩放（保持比赛场次），避免影响其他指标
            adj_team = team_adj.copy(update={
                "xg": float(team_adj.xg) * offense_mult,
                "xga": float(team_adj.xga) * defense_mult,
            })
            label = f"{label_prefix}{metrics.get('team')} 阵型"
            if metrics.get("formation"):
                label += f"·{metrics.get('formation')}"
            mins = metrics.get("min")
            if isinstance(mins, (int, float)):
                label += f"（样本分钟={int(mins)}）"
            label += f" → 进攻×{offense_mult:.2f}，防守×{defense_mult:.2f}"
            fparts.append(label)
            return adj_team

        home_team_adj = _apply_formation_adjust(home_team_adj, "主队·", home_raw_name)
        away_team_adj = _apply_formation_adjust(away_team_adj, "客队·", away_raw_name)
        if fparts:
            history_summary = (history_summary + ("；" if history_summary else "") + "阵型调整：" + "；".join(fparts)).strip()
    except Exception:
        pass

    # 获取定量模型预测（含比分矩阵）—— 使用融合后的 TeamStats

        # —— 联赛参数建议（按请求覆盖 bp_shared_strength / draw_boost_strength，仅当前请求生效） ——
    try:
        raw_payload = await request.json()
        league_preset = raw_payload.get("league_preset")
    except Exception:
        league_preset = None
    _override_bp = None
    _override_draw = None
    if isinstance(league_preset, str):
        lp = league_preset.strip()
        if lp in ("英超/德甲", "prem_bundes"):
            _override_bp, _override_draw = 0.14, 0.08
        elif lp in ("意甲/法甲", "serie_ligue"):
            _override_bp, _override_draw = 0.19, 0.10
        elif lp in ("西甲", "laliga"):
            _override_bp, _override_draw = 0.16, 0.09
    engine_overrides = {}
    if _override_bp is not None:
        engine_overrides["bp_shared_strength"] = float(_override_bp)
    if _override_draw is not None:
        engine_overrides["draw_boost_strength"] = float(_override_draw)
    # 可选：贝叶斯分布扩展开关
    try:
        if getattr(settings, "enable_bayes_engine", False):
            if getattr(settings, "bayes_use_nb", False):
                engine_overrides["use_nb"] = True
            if getattr(settings, "bayes_use_zip", False):
                engine_overrides["use_zip"] = True
    except Exception:
        pass
    _league_note = None
    if _override_bp is not None or _override_draw is not None:
        bpv = _override_bp if _override_bp is not None else getattr(settings, "bp_shared_strength", None)
        drv = _override_draw if _override_draw is not None else getattr(settings, "draw_boost_strength", None)
        try:
            _league_note = f"联赛参数建议：BP_SHARED_STRENGTH={bpv:.2f}, DRAW_BOOST_STRENGTH={drv:.2f}"
        except Exception:
            _league_note = f"联赛参数建议：BP_SHARED_STRENGTH={bpv}, DRAW_BOOST_STRENGTH={drv}"
    
    # 计算模型概率：优先使用贝叶斯泊松引擎（若启用且可用），否则回退默认泊松
    try:
        if getattr(settings, "enable_bayes_engine", False) and BayesPoissonEngine:
            _bp = BayesPoissonEngine(getattr(settings, "bayes_params_key", None))
            model_probs, model_scores, prob_matrix = _bp.predict(
                home_team_adj,
                away_team_adj,
                getattr(match_data, 'league', None),
                overrides=engine_overrides or None
            )
            try:
                history_summary = (history_summary + ("；" if history_summary else "") + "使用贝叶斯泊松引擎").strip()
            except Exception:
                pass
        else:
            model_probs, model_scores, prob_matrix = poisson_engine.calculate_match_probabilities(
                home_team_adj,
                away_team_adj,
                overrides=engine_overrides or None
            )
    except Exception:
        # 回退到默认泊松引擎
        model_probs, model_scores, prob_matrix = poisson_engine.calculate_match_probabilities(
            home_team_adj,
            away_team_adj,
            overrides=engine_overrides or None
        )
    
    # 附加联赛参数建议说明
    try:
        if _league_note:
            history_summary = (history_summary + ("；" if history_summary else "") + _league_note).strip()
    except Exception:
        pass
    try:
        _lg = str(getattr(match_data, "league", None) or getattr(match_data, "league_preset", None) or "").lower()
        _enabled = str(getattr(settings, "gru_enabled_leagues", "意甲,serie a,serie_ligue")).lower().split(",")
        _allow = False
        for w in _enabled:
            w = w.strip()
            if w and w in _lg:
                _allow = True
                break
        _gru = None
        if _allow:
            _load_gru()
            _gru = _predict_with_gru(match_data)
    except Exception:
        _gru = None
    if _gru and isinstance(_gru.get("home_win"), (int, float)):
        _w = float(getattr(settings, "gru_model_weight", 0.50))
        _w = max(0.0, min(1.0, _w))
        adjusted_model_probs = {
            "home_win": _w * float(_gru["home_win"]) + (1.0 - _w) * float(model_probs["home_win"]),
            "draw": _w * float(_gru["draw"]) + (1.0 - _w) * float(model_probs["draw"]),
            "away_win": _w * float(_gru["away_win"]) + (1.0 - _w) * float(model_probs["away_win"]),
        }
    else:
        adjusted_model_probs = model_probs
    
    # 获取市场共识
    if market_engine:
        market_probs, market_trend = market_engine.analyze_market_consensus(match_data.odds_data)
        try:
            _vals = [market_probs.get("home_win"), market_probs.get("draw"), market_probs.get("away_win")]
            if any((v is None) or (not isinstance(v, (int, float))) or (v <= 0) for v in _vals):
                market_probs = {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}
                market_trend = (str(market_trend or "") + "；市场赔率不足，使用均匀分布回退").strip("；")
        except Exception:
            market_probs = {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}
            market_trend = (str(market_trend or "") + "；市场引擎异常，使用均匀分布回退").strip("；")
    else:
        market_probs, market_trend = {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}, "市场引擎不可用"

    # —— 交易所快照融合（可选：仅在已上传或启用回退时参与） ——
    try:
        event_key = f"{match_data.home_team.name}|{match_data.away_team.name}"
        exch = EXCHANGE_CACHE.get(event_key)
        # 可选回退：仅当启用 ENABLE_EXCHANGE_DEFAULT_LOAD 时，才尝试默认路径
        if not exch and getattr(settings, "enable_exchange_default_load", False):
            try:
                exch = load_exchange_features_from_path(settings.exchange_xlsx_path)
            except Exception:
                exch = None
        if exch and isinstance(exch.get("probs"), dict):
            ex_p = exch["probs"]
            ex_w = float(exch.get("liquidity_weight", 0.0) or 0.0)
            # 融合强度：以流动性权重为基础，结合热指/凯差进行温和增强，最高 0.50
            aux = exch.get("aux_metrics") or {}
            try:
                heat_max = float(aux.get("heat_max") or 0.0)
            except Exception:
                heat_max = 0.0
            try:
                kelly_diff = float(aux.get("kelly_diff") or 0.0)
            except Exception:
                kelly_diff = 0.0
            heat_s = max(0.0, min(1.0, heat_max / 100.0)) if heat_max > 1.0 else max(0.0, min(1.0, heat_max))
            kelly_s = max(0.0, min(1.0, kelly_diff / 100.0)) if kelly_diff > 1.0 else max(0.0, min(1.0, kelly_diff))
            base_alpha = max(0.0, min(0.5, 0.5 * ex_w))
            boost = 1.0 + min(0.30, 0.15 * heat_s + 0.15 * kelly_s)
            alpha = max(0.0, min(0.5, base_alpha * boost))
            mh = (1 - alpha) * float(market_probs["home_win"]) + alpha * float(ex_p.get("home_win", market_probs["home_win"]))
            md = (1 - alpha) * float(market_probs["draw"]) + alpha * float(ex_p.get("draw", market_probs["draw"]))
            ma = (1 - alpha) * float(market_probs["away_win"]) + alpha * float(ex_p.get("away_win", market_probs["away_win"]))
            s = max(1e-9, mh + md + ma)
            market_probs = {"home_win": mh / s, "draw": md / s, "away_win": ma / s}
            # 趋势补充
            summ = str(exch.get("summary") or "")
            if summ:
                market_trend = (market_trend + ("；" if market_trend else "") + "交易所融合：" + summ).strip()
    except Exception:
        pass

    # 让盘合理性与异象：模型→让球线映射 + 市场聚合
    expected_handicap = market_engine.compute_fair_handicap(prob_matrix) if market_engine else 0.0
    if market_engine:
        handicap_metrics = market_engine.assess_handicap_anomaly(expected_handicap, match_data.odds_data)
    else:
        handicap_metrics = {
            "market_final_handicap": None,
            "handicap_anomaly_score": None,
            "handicap_anomaly_label": "盘口数据不足",
            "handicap_water_bias": None,
            "handicap_water_diff": None,
            "expected_handicap": expected_handicap,
            "market_initial_handicap": None,
        }

    # 历史市场对照摘要（已移除历史库）
    market_history_summary = ""
    market_history_stats = None

    # —— 模型与市场融合（动态权重） ——
    def _compute_market_weight(company_count: int, stability: float, model_entropy: float, score_peak: float) -> tuple[float, str]:
        """使用标定服务的学习权重；无参数时回退旧启发式。"""
        from ..services.calibration import CalibrationService
        from ..config import settings as _settings
        cs = CalibrationService(_settings.db_path)
        try:
            w, note = cs.compute_market_weight(company_count, stability, model_entropy, score_peak)
            return w, note
        except Exception:
            import math
            w = 0.5
            reasons = []
            if company_count >= 3:
                w += 0.06
                reasons.append("公司数≥3")
            elif company_count == 1:
                w -= 0.06
                reasons.append("公司数=1")
            w += 0.08 * (max(0.0, min(1.0, stability)) - 0.5)
            reasons.append(f"稳定性{stability:.2f}")
            max_ent = math.log(3)
            ent_norm = max(0.0, min(1.0, (model_entropy if model_entropy is not None else 0.0) / max_ent))
            w += 0.08 * (ent_norm - 0.5)
            reasons.append(f"模型不确定性{ent_norm:.2f}")
            w = max(0.30, min(0.70, w))
            return w, "；".join(reasons) + f" → 市场权重={w:.2f}"

    import numpy as np
    eps = 1e-12
    keys = ["home_win", "draw", "away_win"]
    entropy_model = float(-sum(adjusted_model_probs[k] * np.log(adjusted_model_probs[k] + eps) for k in keys))
    score_peak_prob = float(np.max(prob_matrix))
    stability = market_engine.market_stability_score(match_data.odds_data) if market_engine else 0.0

    market_weight, fusion_note = _compute_market_weight(len(match_data.odds_data), stability, entropy_model, score_peak_prob)
    model_weight = 1.0 - market_weight
    # —— 修改：支持几何融合（配置：FUSION_MODE=geometric|linear） ——
    import math
    eps = 1e-12
    if getattr(settings, "fusion_mode", "geometric").lower() == "geometric":
        gm_home = math.exp(model_weight * math.log(max(adjusted_model_probs["home_win"], eps)) + market_weight * math.log(max(market_probs["home_win"], eps)))
        gm_draw = math.exp(model_weight * math.log(max(adjusted_model_probs["draw"], eps)) + market_weight * math.log(max(market_probs["draw"], eps)))
        gm_away = math.exp(model_weight * math.log(max(adjusted_model_probs["away_win"], eps)) + market_weight * math.log(max(market_probs["away_win"], eps)))
        s = gm_home + gm_draw + gm_away
        fused_home, fused_draw, fused_away = gm_home / s, gm_draw / s, gm_away / s
        fusion_note += "；融合模式=几何"
    else:
        fused_home = model_weight * adjusted_model_probs["home_win"] + market_weight * market_probs["home_win"]
        fused_draw = model_weight * adjusted_model_probs["draw"] + market_weight * market_probs["draw"]
        fused_away = model_weight * adjusted_model_probs["away_win"] + market_weight * market_probs["away_win"]
        fusion_note += "；融合模式=线性"

    # —— 新增：融合后平局对齐校准（轻微，默认启用） ——
    try:
        if getattr(settings, "enable_draw_align", True):
            mode = getattr(settings, "draw_align_mode", "market")
            strength = float(getattr(settings, "draw_align_strength", 0.10))
            strength = max(0.0, min(0.30, strength))
            target_draw = float(market_probs.get("draw", fused_draw)) if mode == "market" else float(getattr(settings, "league_draw_base", 0.27))
            # 仅进行微调：朝目标平局移动，随后重新归一化
            new_draw = fused_draw + strength * (target_draw - fused_draw)
            # 重新分配剩余概率保持比例关系
            rest = max(1e-9, fused_home + fused_away)
            if rest > 0:
                scale = max(1e-9, 1.0 - new_draw)
                r_home = fused_home / rest
                r_away = fused_away / rest
                fused_home = r_home * scale
                fused_away = r_away * scale
            fused_draw = max(0.0, min(1.0, new_draw))
            fusion_note += f"；平局对齐={mode} 强度={strength:.2f}"
    except Exception:
        # 若平局对齐失败则回退，不中断流程
        pass

    # —— 新增：融合后概率的后验校准（单调插值），并归一化 ——
    try:
        if getattr(settings, "enable_prob_calibration", True) and ProbCalibrationService:
            _pcs = ProbCalibrationService(settings.db_path)
            _cal = _pcs.apply({"home_win": float(fused_home), "draw": float(fused_draw), "away_win": float(fused_away)})
            fused_home, fused_draw, fused_away = _cal["home_win"], _cal["draw"], _cal["away_win"]
            fusion_note = (fusion_note + "；已应用概率校准").strip()
    except Exception:
        # 若校准失败则安全回退到未校准融合分布
        pass

    # —— 计算校准与一致性指标 ——
    import numpy as np
    eps = 1e-12
    keys = ["home_win", "draw", "away_win"]
    brier_model_vs_market = sum((adjusted_model_probs[k] - market_probs[k]) ** 2 for k in keys)
    kl_model_vs_market = float(sum(
        adjusted_model_probs[k] * np.log((adjusted_model_probs[k] + eps) / (market_probs[k] + eps))
        for k in keys
    ))
    outcome_entropy = float(-sum(adjusted_model_probs[k] * np.log(adjusted_model_probs[k] + eps) for k in keys))
    score_peak_prob = float(np.max(prob_matrix))
    calibration_metrics = {
        "brier_model_vs_market": float(brier_model_vs_market),
        "kl_model_vs_market": float(kl_model_vs_market),
        "outcome_entropy": float(outcome_entropy),
        "score_peak_prob": float(score_peak_prob),
    }
    calibration_notes = (
        "以市场概率为参照，Brier/ KL 距离越小表示一致性越强；"
        "entropy 越小越锐（更果断），score_peak 反映比分矩阵尖锐度。"
    )

    # —— 显示层让球符号翻转（主受让为+、主让为−） ——
    def _flip(val):
        try:
            return -float(val)
        except Exception:
            return None
    disp_expected_handicap = _flip(expected_handicap) if isinstance(expected_handicap, (int, float)) else None
    disp_market_initial_handicap = _flip(handicap_metrics.get("market_initial_handicap")) if isinstance(handicap_metrics.get("market_initial_handicap"), (int, float)) else None
    disp_market_final_handicap = _flip(handicap_metrics.get("market_final_handicap")) if isinstance(handicap_metrics.get("market_final_handicap"), (int, float)) else None

    # —— 冷门因子：识别下狗与爆冷风险分数 ——
    def _clamp01(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0
    # 市场稳定性（越低越容易出冷门）
    market_stability = 0.5
    try:
        market_stability = market_engine.market_stability_score(match_data.odds_data)
    except Exception:
        market_stability = 0.5
    # 市场判定下狗（缺失则按排名）
    mh = market_probs.get("home_win")
    ma = market_probs.get("away_win")
    hr = getattr(match_data.home_team, "ranking", None)
    ar = getattr(match_data.away_team, "ranking", None)
    underdog_side = None
    if isinstance(mh, (int, float)) and isinstance(ma, (int, float)):
        if mh < ma:
            underdog_side = "home"
        elif ma < mh:
            underdog_side = "away"
        else:
            # 平局概率相当时，选择排名更差的一方
            try:
                if isinstance(hr, int) and isinstance(ar, int):
                    underdog_side = "home" if hr > ar else "away"
            except Exception:
                underdog_side = "away"
    else:
        try:
            if isinstance(hr, int) and isinstance(ar, int):
                underdog_side = "home" if hr > ar else "away"
            else:
                underdog_side = "away"
        except Exception:
            underdog_side = "away"

    upset_team = match_data.home_team.name if underdog_side == "home" else match_data.away_team.name
    fused_home = float(fused_home)
    fused_away = float(fused_away)
    underdog_win_prob = fused_home if underdog_side == "home" else fused_away
    market_underdog_prob = mh if underdog_side == "home" else ma
    delta_vs_market = None
    try:
        if isinstance(market_underdog_prob, (int, float)):
            delta_vs_market = underdog_win_prob - float(market_underdog_prob)
    except Exception:
        delta_vs_market = None
    # 盘口方向标记加权
    h_label = str(handicap_metrics.get("handicap_anomaly_label") or "")
    dir_bonus = 0.0
    if "相反" in h_label:
        dir_bonus = 0.25
    elif ("偏深" in h_label) or ("偏浅" in h_label):
        dir_bonus = 0.10
    # 排名差异加权（下狗更差越大）
    rank_gap = 0
    try:
        if underdog_side == "home" and isinstance(hr, int) and isinstance(ar, int):
            rank_gap = max(0, hr - ar)
        elif underdog_side == "away" and isinstance(hr, int) and isinstance(ar, int):
            rank_gap = max(0, ar - hr)
    except Exception:
        rank_gap = 0
    rank_bonus = min(0.15, (rank_gap or 0) / 20.0)
    # 市场低估加权
    underval_adj = 0.0
    if isinstance(delta_vs_market, (int, float)):
        underval_adj = _clamp01(delta_vs_market / 0.30)
    # 终分
    upset_score = _clamp01(underdog_win_prob + 0.5 * underval_adj + dir_bonus + (1.0 - (market_stability or 0.5)) * 0.20 + rank_bonus)
    if upset_score >= 0.65:
        upset_label = "冷门高风险"
    elif upset_score >= 0.45:
        upset_label = "冷门中等"
    else:
        upset_label = "冷门低风险"
    upset_notes = (
        f"下狗：{upset_team}；融合胜率={underdog_win_prob:.1%}；"
        + (f"模型-市场差={delta_vs_market:.1%}；" if isinstance(delta_vs_market, (int, float)) else "")
        + f"盘口判定：{h_label or '—'}；水位：{handicap_metrics.get('handicap_water_bias') or '—'}；"
        + f"市场稳定性={market_stability:.2f}；排名差={rank_gap}"
    )

    # 获取AI分析（加入融合信息）
    if ai_engine:
        analysis = ai_engine.generate_analysis(
            match_data.home_team,
            match_data.away_team,
            adjusted_model_probs,
            model_scores,
            market_probs,
            market_trend,
            history_summary=history_summary,
            market_history_summary=market_history_summary,
            market_history_stats=market_history_stats,
            fused_probs={"home_win": fused_home, "draw": fused_draw, "away_win": fused_away},
            fusion_weights={"model": model_weight, "market": market_weight},
            fusion_notes=fusion_note,
            handicap_expected=disp_expected_handicap,
            handicap_market_initial=disp_market_initial_handicap,
            handicap_market_final=disp_market_final_handicap,
            handicap_anomaly_label=handicap_metrics.get("handicap_anomaly_label"),
            handicap_anomaly_score=handicap_metrics.get("handicap_anomaly_score"),
            handicap_water_bias=handicap_metrics.get("handicap_water_bias"),
        )
    else:
        analysis = "已使用本地融合指标生成分析。"

    # 生成结构化分析（本地，不依赖外部AI）
    if ai_engine:
        analysis_struct = ai_engine.generate_structured_output(
            match_data.home_team,
            match_data.away_team,
            adjusted_model_probs,
            model_scores,
            market_probs,
            market_trend,
            history_summary=history_summary,
            market_history_summary=market_history_summary,
            market_history_stats=market_history_stats,
            fused_probs={"home_win": fused_home, "draw": fused_draw, "away_win": fused_away},
            fusion_weights={"model": model_weight, "market": market_weight},
            fusion_notes=fusion_note,
            handicap_expected=disp_expected_handicap,
            handicap_market_initial=disp_market_initial_handicap,
            handicap_market_final=disp_market_final_handicap,
            handicap_anomaly_label=handicap_metrics.get("handicap_anomaly_label"),
            handicap_anomaly_score=handicap_metrics.get("handicap_anomaly_score"),
            handicap_water_bias=handicap_metrics.get("handicap_water_bias"),
            upset_team=upset_team,
            upset_win_prob=underdog_win_prob,
            upset_score=upset_score,
            upset_label=upset_label,
            upset_delta_vs_market=delta_vs_market,
        )
    else:
        analysis_struct = {
            "format_version": "v1",
            "summary": {
                "overview": "已使用本地融合指标生成结构化分析",
                "fused_probs": {"home_win": fused_home, "draw": fused_draw, "away_win": fused_away},
            },
            "sections": []
        }

    
    
    # 返回前，持久化当前预测（便于历史与回测）
    try:
        _store = PredictionStore(settings.db_path)
        _store.save_prediction(match_data, PredictionResult(
            home_win_prob=adjusted_model_probs["home_win"],
            draw_prob=adjusted_model_probs["draw"],
            away_win_prob=adjusted_model_probs["away_win"],
            likely_scores=model_scores,
            market_home_win_prob=market_probs["home_win"],
            market_draw_prob=market_probs["draw"],
            market_away_win_prob=market_probs["away_win"],
            market_trend=market_trend,
            history_summary=history_summary,
            analysis=analysis,
            market_history_summary=market_history_summary,
            market_history_stats=market_history_stats,
            fused_home_win_prob=fused_home,
            fused_draw_prob=fused_draw,
            fused_away_win_prob=fused_away,
            fusion_weights={"model": model_weight, "market": market_weight},
            fusion_notes=fusion_note,
            expected_handicap=disp_expected_handicap,
            market_final_handicap=disp_market_final_handicap,
            handicap_anomaly_score=handicap_metrics.get("handicap_anomaly_score"),
            handicap_anomaly_label=handicap_metrics.get("handicap_anomaly_label"),
            handicap_water_bias=handicap_metrics.get("handicap_water_bias"),
            score_matrix=prob_matrix.tolist(),
            matrix_goals_max=poisson_engine.max_goals,
            calibration_metrics=calibration_metrics,
            calibration_notes=calibration_notes,
            analysis_source=(getattr(ai_engine, "last_source", None) if ai_engine else "fallback"),
            analysis_struct=analysis_struct,
            upset_team=upset_team,
            upset_win_prob=underdog_win_prob,
            upset_score=upset_score,
            upset_label=upset_label,
            upset_notes=upset_notes,
        ))
    except Exception:
        pass
    
    # 返回最终预测结果（加入比分矩阵与校准指标）
    return PredictionResult(
        home_win_prob=adjusted_model_probs["home_win"],
        draw_prob=adjusted_model_probs["draw"],
        away_win_prob=adjusted_model_probs["away_win"],
        likely_scores=model_scores,
        market_home_win_prob=market_probs["home_win"],
        market_draw_prob=market_probs["draw"],
        market_away_win_prob=market_probs["away_win"],
        market_trend=market_trend,
        history_summary=history_summary,
        analysis=analysis,
        market_history_summary=market_history_summary,
        market_history_stats=market_history_stats,
        fused_home_win_prob=fused_home,
        fused_draw_prob=fused_draw,
        fused_away_win_prob=fused_away,
        fusion_weights={"model": model_weight, "market": market_weight},
        fusion_notes=fusion_note,
        # 盘口合理性与异象
        expected_handicap=disp_expected_handicap,
        market_final_handicap=disp_market_final_handicap,
        handicap_anomaly_score=handicap_metrics.get("handicap_anomaly_score"),
        handicap_anomaly_label=handicap_metrics.get("handicap_anomaly_label"),
        handicap_water_bias=handicap_metrics.get("handicap_water_bias"),
        # 比分概率矩阵与校准指标
        score_matrix=prob_matrix.tolist(),
        matrix_goals_max=poisson_engine.max_goals,
        calibration_metrics=calibration_metrics,
        calibration_notes=calibration_notes,
        analysis_source=(getattr(ai_engine, "last_source", None) if ai_engine else "fallback"),
        analysis_struct=analysis_struct,
        upset_team=upset_team,
        upset_win_prob=underdog_win_prob,
        upset_score=upset_score,
        upset_label=upset_label,
        upset_notes=upset_notes,
    )

@router.post("/admin/roster/import")
async def import_team_roster(file: UploadFile = File(...), team: Optional[str] = Query(None), season: Optional[str] = Query(None)):
    try:
        buf = await file.read()
        import io
        bio = io.BytesIO(buf)
        ext = (file.filename or "").lower()
        import pandas as pd
        if ext.endswith(".xlsx") or ext.endswith(".xls"):
            df = pd.read_excel(bio)
        else:
            try:
                df = pd.read_csv(bio)
            except Exception:
                df = pd.read_csv(io.BytesIO(buf), encoding="utf-8")
        # 统一球队名（若给定 team），读取别名映射
        store = PredictionStore(settings.db_path)
        team_resolved = store.resolve_team_name(team) if team else None
        roster = RosterStore(settings.roster_db_path)
        res = roster.bulk_import(df, team=team_resolved or team, season=season)
        return {"ok": True, "team": team_resolved or team, "season": season, "stats": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/roster")
async def get_team_roster(team: str, season: str):
    try:
        store = PredictionStore(settings.db_path)
        team_resolved = store.resolve_team_name(team)
        roster = RosterStore(settings.roster_db_path)
        items = roster.list_roster(team_resolved, season)
        return {"team": team_resolved, "season": season, "items": items}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/companies")
async def list_companies():
    """返回静态常见公司的列表，避免依赖历史库。"""
    return {"companies": ["365", "皇冠（crown）", "澳门", "易胜博"]}

@router.get("/predictions")
async def list_predictions(limit: int = 50, offset: int = 0, team: Optional[str] = None, has_result: Optional[bool] = None):
    store = PredictionStore(settings.db_path)
    return {"items": store.list_predictions(limit=limit, offset=offset, team=team, has_result=has_result)}

@router.get("/predictions/daily")
async def list_predictions_by_date(date: str, team: Optional[str] = None, has_result: Optional[bool] = None):
    store = PredictionStore(settings.db_path)
    return store.list_predictions_by_date(date, team=team, has_result=has_result)

@router.get("/predictions/calendar")
async def calendar_counts(month: str):
    store = PredictionStore(settings.db_path)
    return store.calendar_counts(month)

@router.get("/predictions/{pid}")
async def get_prediction(pid: int):
    store = PredictionStore(settings.db_path)
    return store.get_prediction(pid)

@router.patch("/predictions/{pid}/result")
async def update_actual(pid: int, payload: Dict):
    hg = int(payload.get("home_goals"))
    ag = int(payload.get("away_goals"))
    store = PredictionStore(settings.db_path)
    return store.update_actual_result(pid, hg, ag)

@router.post("/predictions/{pid}/auto-result")
async def auto_fill_actual(pid: int):
    store = PredictionStore(settings.db_path)
    return store.try_fill_actual_from_matches(pid)

@router.get("/backtest/summary")
async def backtest_summary():
    store = PredictionStore(settings.db_path)
    return store.backtest_summary()

@router.get("/backtest/calibration")
async def backtest_calibration():
    store = PredictionStore(settings.db_path)
    return store.backtest_calibration_by_class()

@router.get("/admin/teams")
async def list_team_season_stats(team: Optional[str] = None, season: Optional[str] = None, league: Optional[str] = None, limit: int = 50, offset: int = 0):
    store = PredictionStore(settings.db_path)
    return {"items": store.list_team_season_stats(team=team, season=season, league=league, limit=limit, offset=offset)}

@router.post("/admin/teams")
async def upsert_team_season_stats(payload: TeamSeasonStats):
    try:
        store = PredictionStore(settings.db_path)
        res = store.upsert_team_season_stats(payload)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/admin/matches")
async def insert_match_manual(payload: AdminMatch):
    try:
        store = PredictionStore(settings.db_path)
        # 优先写入原库 matches_data（字段对齐），若不存在则回退到 matches_manual
        res = store.insert_match_original(payload)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/admin/config")
async def get_runtime_config():
    try:
        return {
            "require_ai_api": bool(getattr(settings, "require_ai_api", False)),
            "gru_enabled_leagues": str(getattr(settings, "gru_enabled_leagues", "意甲,serie a,serie_ligue")),
            "gru_model_weight": float(getattr(settings, "gru_model_weight", 0.50)),
            "ai_key_present": bool(getattr(settings, "deepseek_api_key", "") != ""),
            "ai_api_base": str(getattr(settings, "deepseek_api_base", "")),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/admin/config")
async def set_runtime_config(request: Request):
    try:
        payload = await request.json()
        if "require_ai_api" in payload:
            v = payload.get("require_ai_api")
            if isinstance(v, str):
                s = v.strip().lower()
                setattr(settings, "require_ai_api", s in ("true", "1", "yes", "是"))
            elif isinstance(v, bool):
                setattr(settings, "require_ai_api", bool(v))
        if "gru_enabled_leagues" in payload:
            v = payload.get("gru_enabled_leagues")
            if isinstance(v, str):
                setattr(settings, "gru_enabled_leagues", v)
        if "gru_model_weight" in payload:
            v = payload.get("gru_model_weight")
            try:
                w = float(v)
            except Exception:
                w = float(getattr(settings, "gru_model_weight", 0.50))
            w = max(0.0, min(1.0, w))
            setattr(settings, "gru_model_weight", w)
        return await get_runtime_config()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/admin/ai-status")
async def get_ai_status():
    try:
        enabled = False
        if AIEngine:
            try:
                _ai = AIEngine()
                enabled = bool(getattr(_ai, "enabled", False))
            except Exception:
                enabled = False
        return {
            "require_ai_api": bool(getattr(settings, "require_ai_api", False)),
            "ai_key_present": bool(getattr(settings, "deepseek_api_key", "") != ""),
            "ai_enabled": enabled,
            "ai_api_base": str(getattr(settings, "deepseek_api_base", "")),
            "ai_key_length": (len(getattr(settings, "deepseek_api_key", "")) if getattr(settings, "deepseek_api_key", "") else 0),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/admin/exchange-snapshot/upload")
async def upload_exchange_snapshot(
    file: UploadFile = File(...),
    home: Optional[str] = Query(None),
    away: Optional[str] = Query(None),
    event_key: Optional[str] = Query(None),
    sheet: Optional[str] = Query(None)
):
    """
    上传必发/交易所快照（Excel），即时解析并缓存该场比赛的交易所特征。
    - 通过 query 传入主客队以建立缓存键（若未传则仅返回解析结果）。
    - 支持 sheet 名（或索引字符串）选择。
    """
    try:
        buf = await file.read()
        exch = load_exchange_features_from_bytes(buf, sheet=sheet)
        # —— 清洗 NaN/Inf，避免 JSON 编码错误 ——
        import math
        def _clean_json(obj):
            try:
                if obj is None:
                    return None
                if isinstance(obj, (int, float)):
                    v = float(obj)
                    return v if math.isfinite(v) else None
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, dict):
                    return {str(k): _clean_json(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [ _clean_json(v) for v in obj ]
                return obj
            except Exception:
                return None
        exch = _clean_json(exch)
        # 解析球队规范名（若提供），构建事件键
        ek = event_key
        try:
            if not ek and (home or away):
                store = PredictionStore(settings.db_path)
                h = store.resolve_team_name(home) if home else None
                a = store.resolve_team_name(away) if away else None
                ek = f"{h or (home or '')}|{a or (away or '')}".strip("|")
        except Exception:
            if not ek and (home or away):
                ek = f"{home or ''}|{away or ''}".strip("|")

        if ek:
            EXCHANGE_CACHE[ek] = exch
        return {"ok": True, "event_key": ek, "features": exch}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/admin/team-metrics/template")
async def team_metrics_template(
    format: str = Query("xlsx", regex="^(xlsx|csv)$"),
    style: str = Query("compact", regex="^(compact|standard)$")
):
    import pandas as pd
    # 标准列（与存储表一致）
    standard_cols = [
        "team","season","date","league","games_played","points","wins","draws","losses",
        "goals_for","goals_against","xg","npxg","xga","npxga","xpxgd","ppda","oppda","dc","odc","xpts"
    ]
    # 紧凑列（符合图片表头；另加 Season 字段）
    compact_cols = [
        "№","Team","Season","M","W","D","L","G","GA","PTS","xG","NPxG","xGA","NPxGA","NPxGD","PPDA","OPPDA","DC","ODC","xPTS"
    ]
    cols = compact_cols if style == "compact" else standard_cols
    df = pd.DataFrame([], columns=cols)
    if format == "xlsx":
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            sheet_name = "team_metrics_template_compact" if style == "compact" else "team_metrics_template"
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        bio.seek(0)
        fname = "team_metrics_template_compact.xlsx" if style == "compact" else "team_metrics_template.xlsx"
        headers = {"Content-Disposition": f"attachment; filename={fname}"}
        return StreamingResponse(bio, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)
    else:
        sio = io.StringIO()
        df.to_csv(sio, index=False)
        fname = "team_metrics_template_compact.csv" if style == "compact" else "team_metrics_template.csv"
        headers = {"Content-Disposition": f"attachment; filename={fname}"}
        return StreamingResponse(iter([sio.getvalue()]), media_type="text/csv", headers=headers)

@router.post("/admin/team-metrics/import")
async def import_team_metrics(file: UploadFile = File(...)):
    try:
        import pandas as pd
        # 读取文件到 DataFrame
        ext = (file.filename or "").lower()
        buf = await file.read()
        import io
        bio = io.BytesIO(buf)
        if ext.endswith(".xlsx") or ext.endswith(".xls"):
            df = pd.read_excel(bio)
        else:
            try:
                df = pd.read_csv(bio)
            except Exception:
                df = pd.read_csv(io.BytesIO(buf), encoding="utf-8")
        df.columns = [str(c).strip().lower() for c in df.columns]
        # 别名映射与列填充（略）
        alias = {
            "team": "team", "season": "season", "date": "date", "league": "league",
            # 紧凑表头映射
            "m": "games_played", "w": "wins", "d": "draws", "l": "losses",
            "pts": "points", "g": "goals_for", "ga": "goals_against",
            # 扩展与指标
            "xg": "xg", "npxg": "npxg", "xga": "xga", "npxga": "npxga", "npxgd": "xpxgd", "xpxg_diff": "xpxgd",
            "ppda": "ppda", "oppda": "oppda", "dc": "dc", "odc": "odc", "xpts": "xpts",
            # 常见别名
            "gf": "goals_for"
        }
        for k, v in alias.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]
        cols = [
            "team","season","date","league","games_played","points","wins","draws","losses",
            "goals_for","goals_against","xg","npxg","xga","npxga","xpxgd","ppda","oppda","dc","odc","xpts"
        ]
        store = PredictionStore(settings.db_path)
        # 覆盖模式：先删除文件中包含的赛季数据（保留现有逻辑）
        deleted = 0
        try:
            if "season" in df.columns:
                seasons = list(set([str(s) for s in df["season"].dropna().astype(str).tolist()]))
                if seasons:
                    deleted = store.delete_team_metrics_daily_by_seasons(seasons)
        except Exception:
            deleted = 0
        imported = 0
        errors = []
        for i, row in df.iterrows():
            try:
                rec = {}
                for c in cols:
                    val = row[c] if c in df.columns else None
                    try:
                        if pd.isna(val):
                            val = None
                    except Exception:
                        pass
                    rec[c] = val
                # 统一数值字段类型与空白处理，避免空字符串落库影响前端填充
                try:
                    int_fields = [
                        "games_played","wins","draws","losses","goals_for","goals_against"
                    ]
                    float_fields = [
                        "points","xg","npxg","xga","npxga","xpxgd","ppda","oppda","dc","odc","xpts"
                    ]
                    for f in int_fields:
                        v = rec.get(f)
                        if isinstance(v, str):
                            s = v.strip()
                            rec[f] = int(s) if s != '' else None
                        elif v is None:
                            rec[f] = None
                        else:
                            try:
                                rec[f] = int(v)
                            except Exception:
                                rec[f] = None
                    for f in float_fields:
                        v = rec.get(f)
                        if isinstance(v, str):
                            s = v.strip()
                            rec[f] = float(s) if s != '' else None
                        elif v is None:
                            rec[f] = None
                        else:
                            try:
                                rec[f] = float(v)
                            except Exception:
                                rec[f] = None
                except Exception:
                    # 规范化失败不阻断导入，保持原值
                    pass
                if not rec.get("date"):
                    try:
                        rec["date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
                    except Exception:
                        rec["date"] = None
                # 统一队名为规范名
                try:
                    rec["team"] = store.resolve_team_name(rec.get("team"), rec.get("league"))
                except Exception:
                    pass
                if not rec.get("team") or not rec.get("season") or not rec.get("date"):
                    raise ValueError("缺少必填字段：team/season/date")
                store.upsert_team_metrics_daily(rec)
                imported += 1
            except Exception as e:
                errors.append({"row": int(i), "error": str(e)})
        return {"imported": imported, "errors": errors, "overwritten": deleted}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/teams/metrics")
async def get_team_metrics(team: str, season: Optional[str] = None):
    store = PredictionStore(settings.db_path)
    # 统一队名：别名解析，未命中时回退原始值
    try:
        team_resolved = store.resolve_team_name(team)
    except Exception:
        team_resolved = team
    res = store.get_team_latest_metrics(team_resolved, season)
    if not res and team_resolved != team:
        # 别名解析未能匹配时，尝试用原名查询一次
        res = store.get_team_latest_metrics(team, season)
    if not res:
        raise HTTPException(status_code=404, detail="未找到球队指标")
    return res

@router.post("/admin/team-aliases/import")
async def import_team_aliases(file: UploadFile = File(...)):
    try:
        store = PredictionStore(settings.db_path)
        buf = await file.read()
        import io
        bio = io.BytesIO(buf)
        import pandas as pd
        ext = (file.filename or "").lower()
        if ext.endswith(".xlsx") or ext.endswith(".xls"):
            df = pd.read_excel(bio)
        else:
            try:
                df = pd.read_csv(bio)
            except Exception:
                df = pd.read_csv(io.BytesIO(buf), encoding="utf-8")
        # 统一列名小写并去空格
        df.columns = [str(c).strip().lower() for c in df.columns]
        cols = set(df.columns)
        # 识别规范名与别名列
        canonical_candidates = ["canonical", "canonical_name", "中文名", "中文", "name_cn", "cn", "规范名", "中文队名"]
        alias_candidates = ["alias", "alias_name", "英文名", "英文", "name_en", "en", "简称", "简写", "别名", "队名英文", "队名简写"]
        league_col = None
        lang_col = None
        # 找规范列
        canonical_col = None
        for c in canonical_candidates:
            if c.lower() in cols:
                canonical_col = c.lower()
                break
        # 找别名列（可能多个）
        alias_cols = []
        for a in alias_candidates:
            if a.lower() in cols:
                alias_cols.append(a.lower())
        # 可选联赛、语言列
        for cand in ["league", "联赛", "联赛名称", "league_name"]:
            if cand.lower() in cols:
                league_col = cand.lower(); break
        for cand in ["lang", "语言"]:
            if cand.lower() in cols:
                lang_col = cand.lower(); break
        if canonical_col is None or not alias_cols:
            raise HTTPException(status_code=400, detail="映射文件缺少规范名或别名列")
        items = []
        for _, row in df.iterrows():
            cn = str(row[canonical_col]).strip() if row.get(canonical_col) is not None else ""
            if not cn:
                continue
            lg = str(row[league_col]).strip() if (league_col and row.get(league_col) is not None) else None
            ln = str(row[lang_col]).strip() if (lang_col and row.get(lang_col) is not None) else None
            for ac in alias_cols:
                al = row.get(ac)
                if al is None:
                    continue
                al = str(al).strip()
                if not al:
                    continue
                items.append({"canonical_name": cn, "alias_name": al, "league": lg, "lang": ln})
        res = store.bulk_upsert_team_aliases(items)
        return {"imported": res.get("imported", 0)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/admin/odds/excel")
async def import_odds_excel(
    file: UploadFile = File(...),
    sheet: Optional[str] = Query(None),
    sheet_handicap: Optional[str] = Query(None),
    sheet_1x2: Optional[str] = Query(None)
):
    try:
        import pandas as pd, io, re
        buf = await file.read()
        bio = io.BytesIO(buf)
        def read_sheet(sn):
            try:
                sn_idx = int(sn) if isinstance(sn, str) and sn.isdigit() else sn
            except Exception:
                sn_idx = sn
            try:
                return pd.read_excel(bio, sheet_name=(sn_idx if sn_idx is not None else 0), engine="openpyxl")
            except Exception:
                return pd.read_excel(io.BytesIO(buf), sheet_name=(sn_idx if sn_idx is not None else 0))
        def to_lower(df):
            df.columns = [str(c).strip().lower() for c in df.columns]
            return df
        def pick(cols, cands, kw=None):
            s = set(cols)
            for c in cands:
                if c in s:
                    return c
            if kw:
                for c in cols:
                    t = str(c)
                    if all(k in t for k in kw):
                        return c
            return None
        def to_f(v):
            try:
                if v is None: return 0.0
                s = str(v).strip()
                if s == '' or s == '-': return 0.0
                return float(s)
            except Exception:
                return 0.0
        def norm_hcap(t:str)->str:
            s = str(t or "")
            pos = "+" if ("受" in s or "受让" in s) else "-"
            m = None
            for k,v in {"平手":0.0,"平手/半球":0.25,"半球":0.5,"半球/一球":0.75,"一球":1.0,"一球/球半":1.25,"球半":1.5,"球半/两球":1.75,"两球":2.0,"两球/两球半":2.25,"两球半":2.5}.items():
                if k in s: m = v; break
            if m is None:
                r = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
                m = float(r.group(1)) if r else 0.0
            if m==0.0: return "0"
            return f"{pos}{m}".replace("+-","-")
        if sheet_1x2 or sheet_handicap:
            try:
                df_ah = to_lower(read_sheet(sheet_handicap or 0))
            except Exception:
                df_ah = to_lower(read_sheet(0))
            try:
                df_1x2 = to_lower(read_sheet(sheet_1x2 or 0))
            except Exception:
                df_1x2 = None
        else:
            df_ah = to_lower(read_sheet(sheet or 0))
            df_1x2 = None
        ah_cols = df_ah.columns if df_ah is not None else []
        try:
            df_ah = df_ah.copy()
        except Exception:
            pass
        comp_ah = pick(ah_cols, ["公司","bookmaker","company","机构","庄家"], kw=["公司"]) or pick(ah_cols, ["公司","bookmaker","company","机构","庄家"]) 
        ihcap = pick(ah_cols, ["初始盘口","初盘","初盘盘口"], kw=["初","盘"]) 
        fhcap = pick(ah_cols, ["最新盘口","终盘","终盘盘口"], kw=["新","盘"]) 
        def num_ratio(c):
            try:
                s = pd.to_numeric(df_ah[c], errors="coerce")
                n = int(s.notna().sum())
                d = int(len(s))
                return (n / d) if d else 0.0
            except Exception:
                return 0.0
        def pick_tokens(cols, must, any_tokens=None, prefer_range=None):
            s = set(cols)
            ranked = []
            for c in s:
                name = str(c)
                if all(t in name for t in must):
                    score = 1.0
                    if any_tokens:
                        score += sum(0.1 for t in any_tokens if t in name)
                    nr = num_ratio(c)
                    if prefer_range and nr >= 0.4:
                        try:
                            vals = pd.to_numeric(df_ah[c], errors="coerce")
                            vals = vals.dropna()
                            if len(vals) > 0:
                                med = float(vals.median())
                                if prefer_range[0] <= med <= prefer_range[1]:
                                    score += 0.2
                        except Exception:
                            pass
                    ranked.append((score, c))
            if not ranked:
                return None
            ranked.sort(key=lambda x: -x[0])
            return ranked[0][1]
        ihh = pick(ah_cols, ["初盘主","初主","主盘","home_open","主水(初)","主"], kw=["初","主"]) or pick_tokens(ah_cols, ["初","主"], any_tokens=["水","赔","队"], prefer_range=(0.6, 3.0)) 
        iha = pick(ah_cols, ["初盘客","初客","客盘","away_open","客水(初)","客"], kw=["初","客"]) or pick_tokens(ah_cols, ["初","客"], any_tokens=["水","赔","队"], prefer_range=(0.6, 3.0)) 
        fhh = pick(ah_cols, ["最新主","终盘主","主水","home_close","主水(新)","主"], kw=["新","主"]) or pick_tokens(ah_cols, ["新","主"], any_tokens=["终","水","赔","队"], prefer_range=(0.6, 3.0)) 
        fha = pick(ah_cols, ["最新客","终盘客","客水","away_close","客水(新)","客"], kw=["新","客"]) or pick_tokens(ah_cols, ["新","客"], any_tokens=["终","水","赔","队"], prefer_range=(0.6, 3.0)) 
        if not (ihcap and fhcap):
            try:
                def token_ratio(c):
                    s = df_ah[c].astype(str)
                    hits = s.str.contains("受|半球|平手|球半|两球|盘口", na=False).sum()
                    return hits / max(1, len(s))
                scored = sorted([(c, token_ratio(c)) for c in df_ah.columns], key=lambda x: -x[1])
                ihcap = ihcap or (scored[0][0] if scored and scored[0][1] >= 0.2 else ihcap)
                fhcap = fhcap or (scored[1][0] if len(scored) > 1 and scored[1][1] >= 0.2 else fhcap)
            except Exception:
                pass
        if comp_ah:
            try:
                header_tokens = {"公司","地域","全部亚盘","全部欧赔","机构","bookmaker","company"}
                col_name_token = str(comp_ah).strip().lower()
                def _is_header_like(v):
                    try:
                        s = str(v).strip()
                    except Exception:
                        s = ""
                    if not s or s.lower() == "nan":
                        return True
                    if s.strip().lower() == col_name_token:
                        return True
                    if s in header_tokens:
                        return True
                    return False
                df_ah = df_ah[~df_ah[comp_ah].apply(_is_header_like)]
            except Exception:
                pass
        if not (ihh and iha and fhh and fha):
            try:
                idx = {col: i for i, col in enumerate(df_ah.columns)}
                numeric_cols = [c for c in df_ah.columns if num_ratio(c) >= 0.6]
                def pick_after(base_col, k=2):
                    if base_col is None:
                        return []
                    base_idx = idx.get(base_col, -1)
                    cands = [c for c in numeric_cols if idx.get(c, 9999) > base_idx]
                    return cands[:k]
                ih_pair = pick_after(ihcap, 4)
                fh_pair = pick_after(fhcap, 4)
                def choose_pair(cands, is_initial=True):
                    if not cands:
                        return []
                    scored = []
                    for c in cands:
                        name = str(c)
                        score = 0.0
                        score += 0.3 if ("主" in name) else 0.0
                        score += 0.3 if ("客" in name) else 0.0
                        score += 0.2 if ("水" in name or "赔" in name) else 0.0
                        score += 0.2 if ("队" in name) else 0.0
                        if is_initial:
                            score += 0.3 if ("初" in name) else 0.0
                        else:
                            score += 0.3 if (("新" in name) or ("终" in name)) else 0.0
                        try:
                            vals = pd.to_numeric(df_ah[c], errors="coerce").dropna()
                            if len(vals) > 0:
                                med = float(vals.median())
                                if 0.6 <= med <= 3.0:
                                    score += 0.2
                        except Exception:
                            pass
                        scored.append((score, c))
                    scored.sort(key=lambda x: -x[0])
                    top = [c for _, c in scored[:2]]
                    return top
                ih_sel = choose_pair(ih_pair, True)
                fh_sel = choose_pair(fh_pair, False)
                if len(ih_sel) >= 2:
                    ihh = ihh or ih_sel[0]
                    iha = iha or ih_sel[1]
                if len(fh_sel) >= 2:
                    fhh = fhh or fh_sel[0]
                    fha = fha or fh_sel[1]
                if (not ihh or not iha) and numeric_cols:
                    sel = choose_pair(numeric_cols, True)
                    if len(sel) >= 2:
                        ihh = ihh or sel[0]
                        iha = iha or sel[1]
                if (not fhh or not fha) and numeric_cols:
                    sel = choose_pair(numeric_cols, False)
                    if len(sel) >= 2:
                        fhh = fhh or sel[0]
                        fha = fha or sel[1]
            except Exception:
                pass
        if not comp_ah:
            try:
                idx = {col: i for i, col in enumerate(df_ah.columns)}
                base_idx = idx.get(ihcap, 9999)
                for c in df_ah.columns:
                    if idx.get(c, 9999) < base_idx:
                        try:
                            s = pd.to_numeric(df_ah[c], errors="coerce")
                            if s.notna().sum() == 0:
                                comp_ah = c
                                break
                        except Exception:
                            comp_ah = c
                            break
            except Exception:
                pass
        if df_1x2 is not None:
            one_cols = df_1x2.columns
            comp_1x2 = pick(one_cols, ["公司","bookmaker","company","机构","庄家"], kw=["公司"]) or pick(one_cols, ["公司","bookmaker","company","机构","庄家"]) 
            ih = pick(one_cols, ["初始主胜","初主胜","主胜(初)","home_open"], kw=["初","主"]) 
            idr = pick(one_cols, ["初始平","初平","平(初)","draw_open"], kw=["初","平"]) 
            ia = pick(one_cols, ["初始客胜","初客胜","客胜(初)","away_open"], kw=["初","客"]) 
            fh = pick(one_cols, ["最新主胜","终盘主胜","主胜(新)","home_close"], kw=["新","主"]) 
            fdr = pick(one_cols, ["最新平","终盘平","平(新)","draw_close"], kw=["新","平"]) 
            fa = pick(one_cols, ["最新客胜","终盘客胜","客胜(新)","away_close"], kw=["新","客"]) 
        else:
            comp_1x2 = ih = idr = ia = fh = fdr = fa = None
        rows = {}
        if df_ah is not None:
            for _, r in df_ah.iterrows():
                c = str(r.get(comp_ah) or "").strip() or "Unknown"
                try:
                    import re
                    c = re.sub(r"[^\w\u4e00-\u9fa5]+", "", c)
                except Exception:
                    pass
                obj = rows.get(c) or {"company": c}
                obj.update({
                    "initial_handicap": norm_hcap(r.get(ihcap)),
                    "final_handicap": norm_hcap(r.get(fhcap)),
                    "initial_handicap_home_odds": to_f(r.get(ihh)),
                    "initial_handicap_away_odds": to_f(r.get(iha)),
                    "final_handicap_home_odds": to_f(r.get(fhh)),
                    "final_handicap_away_odds": to_f(r.get(fha)),
                })
                rows[c] = obj
        if df_1x2 is not None:
            for _, r in df_1x2.iterrows():
                c = str(r.get(comp_1x2) or "").strip() or "Unknown"
                obj = rows.get(c) or {"company": c}
                obj.update({
                    "initial_home_win": to_f(r.get(ih)),
                    "initial_draw": to_f(r.get(idr)),
                    "initial_away_win": to_f(r.get(ia)),
                    "final_home_win": to_f(r.get(fh)),
                    "final_draw": to_f(r.get(fdr)),
                    "final_away_win": to_f(r.get(fa)),
                })
                rows[c] = obj
        odds = []
        for c, obj in rows.items():
            for k in [
                "initial_home_win","initial_draw","initial_away_win",
                "final_home_win","final_draw","final_away_win",
                "initial_handicap","final_handicap",
                "initial_handicap_home_odds","initial_handicap_away_odds",
                "final_handicap_home_odds","final_handicap_away_odds"
            ]:
                if k not in obj:
                    obj[k] = 0 if "odds" in k or "win" in k or "draw" in k else "0"
            odds.append(obj)
        # 清洗无效/标题行
        try:
            header_tokens = {"公司","地域","全部亚盘","全部欧赔","机构","bookmaker","company",""}
            def _is_valid(o):
                cn = str(o.get("company") or "").strip()
                if (not cn) or (cn.lower() == "nan") or (cn in header_tokens):
                    return False
                has_line = str(o.get("initial_handicap") or "0") != "0" or str(o.get("final_handicap") or "0") != "0"
                has_odds = any(float(o.get(k) or 0) > 0 for k in [
                    "initial_home_win","initial_draw","initial_away_win",
                    "final_home_win","final_draw","final_away_win",
                    "initial_handicap_home_odds","initial_handicap_away_odds",
                    "final_handicap_home_odds","final_handicap_away_odds"
                ])
                return has_line or has_odds
            odds = [o for o in odds if _is_valid(o)]
        except Exception:
            pass
        cols_map = {
            "company": comp_ah,
            "initial_handicap": ihcap,
            "final_handicap": fhcap,
            "initial_handicap_home_odds": ihh,
            "initial_handicap_away_odds": iha,
            "final_handicap_home_odds": fhh,
            "final_handicap_away_odds": fha,
        }
        if df_1x2 is not None:
            cols_map.update({
                "initial_home_win": ih,
                "initial_draw": idr,
                "initial_away_win": ia,
                "final_home_win": fh,
                "final_draw": fdr,
                "final_away_win": fa,
            })
        return {"ok": True, "odds_data": odds, "columns": cols_map}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
