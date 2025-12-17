from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from ..models.team import TeamStats
from ..engines.poisson_engine import PoissonEngine, ScoreProbability
from ..services.bayes_params_store import BayesParamsStore


class BayesPoissonEngine:
    """封装：使用分层贝叶斯团队参数计算 mu_home/mu_away，
    并调用 PoissonEngine（支持 NB/ZIP/BP/DC/DrawBoost）。
    """

    def __init__(self, params_key: Optional[str] = None):
        self.store = BayesParamsStore(params_key)
        self.poisson = PoissonEngine()
        self.params = self.store.load() or {}

    def _team_param(self, team_name: str, league: Optional[str] = None) -> Tuple[float, float]:
        # 返回 (attack, defense) 的对数尺度参数，缺失时回退 0
        att_map = self.params.get("team_attack", {})
        def_map = self.params.get("team_defense", {})
        # 联赛分层可用键 "{league}:{team}"，若无则用纯队名
        key_l = f"{league}:{team_name}" if league else team_name
        att = float(att_map.get(key_l, att_map.get(team_name, 0.0)))
        deff = float(def_map.get(key_l, def_map.get(team_name, 0.0)))
        return att, deff

    def _scalars(self) -> Tuple[float, float, float]:
        # 返回 log_base, home_advantage_log, tempo_log（若无则为 0）
        log_base = float(self.params.get("log_base", 0.0))
        ha_log = float(self.params.get("home_advantage_log", 0.0))
        tempo_log = float(self.params.get("tempo_log", 0.0))
        return log_base, ha_log, tempo_log

    def predict(self, home_team: TeamStats, away_team: TeamStats, league: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, float], List[ScoreProbability], np.ndarray]:
        overrides = overrides or {}
        # 计算贝叶斯期望进球（log 链接）：
        # log(mu_h) = log_base + tempo_log + ha_log + att_h - def_a
        # log(mu_a) = log_base + tempo_log + att_a - def_h
        att_h, def_h = self._team_param(home_team.name, league)
        att_a, def_a = self._team_param(away_team.name, league)
        log_base, ha_log, tempo_log = self._scalars()
        mu_home = float(np.exp(log_base + tempo_log + ha_log + att_h - def_a))
        mu_away = float(np.exp(log_base + tempo_log + att_a - def_h))

        # 分布与相关性/平局参数（若训练后保存）
        if "nb_k" in self.params:
            overrides.setdefault("use_nb", True)
            overrides.setdefault("nb_k", float(self.params.get("nb_k", 10.0)))
        if "zip_pi_home" in self.params or "zip_pi_away" in self.params:
            overrides.setdefault("use_zip", True)
            overrides.setdefault("zip_pi_home", float(self.params.get("zip_pi_home", 0.0)))
            overrides.setdefault("zip_pi_away", float(self.params.get("zip_pi_away", 0.0)))
        if "bp_shared_strength" in self.params:
            overrides.setdefault("bp_shared_strength", float(self.params.get("bp_shared_strength", 0.15)))
        if "draw_boost_strength" in self.params:
            overrides.setdefault("draw_boost_strength", float(self.params.get("draw_boost_strength", 0.08)))

        overrides["mu_home"] = mu_home
        overrides["mu_away"] = mu_away

        return self.poisson.calculate_match_probabilities(home_team, away_team, overrides)