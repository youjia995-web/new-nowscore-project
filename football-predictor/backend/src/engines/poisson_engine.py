import numpy as np
from scipy.stats import poisson
from typing import List, Tuple, Dict, Optional
from ..models.match import TeamStats, ScoreProbability
from ..config import settings

class PoissonEngine:
    def __init__(self, total_teams=20, strength_influence_factor=0.3, ranking_weight=0.2):
        self.max_goals = 5  # 默认上限（会在计算时自适应）
        self.total_teams = total_teams
        self.strength_influence_factor = strength_influence_factor
        self.ranking_weight = ranking_weight
        self.avg_base_score = (total_teams + 1) / 2
        # 自适应设置
        self.cap_min = 5
        self.cap_max = 10
        self.tail_mass = 0.999  # 目标覆盖质量（CDF阈值）

    # —— 新增：矩阵归一化与低比分修正/平局加强 —— 
    def _renorm_matrix(self, mat: np.ndarray) -> np.ndarray:
        s = float(np.sum(mat))
        if s <= 0:
            return mat
        return mat / s

    def _apply_dc_correlation(self, mat: np.ndarray, mu_h: float, mu_a: float, rho: float) -> np.ndarray:
        """Dixon-Coles 低比分相关性：对(0,0),(0,1),(1,0),(1,1)进行修正后归一化。"""
        # 保护性：防止极端 rho
        r0 = float(max(-0.35, min(0.35, rho)))
        if getattr(settings, "enable_dc_rho_dynamic", False):
            denom = 1.0 + max(0.0, mu_h) + max(0.0, mu_a)
            r = float(max(-0.35, min(0.35, r0 / denom)))
        else:
            r = r0
        tau = np.ones_like(mat)
        if mat.shape[0] >= 2 and mat.shape[1] >= 2:
            tau[0, 0] = 1.0 - (mu_h + mu_a) * r
            tau[0, 1] = 1.0 + mu_h * r
            tau[1, 0] = 1.0 + mu_a * r
            tau[1, 1] = 1.0 - r
        adj = mat * tau
        return self._renorm_matrix(adj)

    def _apply_draw_boost(self, mat: np.ndarray, mu_h: float, mu_a: float, strength: float) -> np.ndarray:
        """可选平局加强：对对角线比分按均衡度增强，随后归一化。"""
        g = float(max(0.0, min(0.30, strength)))  # 上限保护
        balance = float(np.exp(-abs(mu_h - mu_a)))  # 越均衡，越增强
        boost = 1.0 + g * balance
        # 对角线增强
        i_max = min(mat.shape[0], mat.shape[1])
        for i in range(i_max):
            mat[i, i] *= boost
        return self._renorm_matrix(mat)

    # —— 新增：双变量泊松（共享成分） ——
    def _bivariate_poisson_matrix(self, mu_h: float, mu_a: float, shared_strength: float) -> np.ndarray:
        """构建具有共享成分C的双变量泊松比分矩阵。
        X = Y1 + C, Y = Y2 + C，其中 Y1~Poi(l1), Y2~Poi(l2), C~Poi(lc)。
        设 lc = min(mu_h, mu_a) * shared_strength * balance，balance=exp(-|mu_h-mu_a|)。
        保持边缘期望：l1 = mu_h - lc，l2 = mu_a - lc。
        """
        # 共享强度与均衡度
        ss = float(max(0.0, min(1.0, shared_strength)))
        balance = float(np.exp(-abs(mu_h - mu_a)))
        lc_raw = float(max(0.0, min(mu_h, mu_a))) * ss * balance
        # 保护：共享不超过边缘的 60%
        lc = float(max(0.0, min(lc_raw, 0.60 * float(min(mu_h, mu_a)))))
        l1 = float(max(0.0, mu_h - lc))
        l2 = float(max(0.0, mu_a - lc))

        # 自适应上限（与独立路径一致）
        cap_h = self._adaptive_goal_cap(mu_h)
        cap_a = self._adaptive_goal_cap(mu_a)
        self.max_goals = max(cap_h, cap_a)

        # 常数前因子
        try:
            front = float(np.exp(-(l1 + l2 + lc)))
        except Exception:
            front = 0.0

        # 预计算阶乘与幂，避免重复计算
        max_g = self.max_goals
        # 阶乘缓存
        fact = np.array([np.math.factorial(i) for i in range(max_g + 1)], dtype=float)
        # 幂缓存（含k项）
        pow_l1 = np.array([l1 ** i for i in range(max_g + 1)], dtype=float)
        pow_l2 = np.array([l2 ** j for j in range(max_g + 1)], dtype=float)
        pow_lc = np.array([lc ** k for k in range(max_g + 1)], dtype=float)

        mat = np.zeros((max_g + 1, max_g + 1))
        if front <= 0.0:
            return mat
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                s = 0.0
                m = min(i, j)
                # Σ_k=0..m (l1^{i-k}/(i-k)!)(l2^{j-k}/(j-k)!)(lc^k/k!)
                for k in range(m + 1):
                    s += (pow_l1[i - k] / fact[i - k]) * (pow_l2[j - k] / fact[j - k]) * (pow_lc[k] / fact[k])
                mat[i, j] = front * s
        return self._renorm_matrix(mat)

    def _dynamic_shared_strength(self, mu_h: float, mu_a: float, base: float) -> float:
        """根据总预期进球对共享强度进行动态修正：低总球→更强相关，高总球→更弱。
        ss_eff = base * scale(total; center, span, amp)，并限制到 [0.05, 0.35]。
        """
        try:
            center = float(getattr(settings, "bp_total_center", 2.6))
            span = float(getattr(settings, "bp_total_span", 1.0))
            amp = float(getattr(settings, "bp_total_amp", 0.25))
            total = float(mu_h + mu_a)
            # 线性缩放并保护上下限
            scale = 1.0 + amp * ((center - total) / max(span, 1e-6))
            scale = max(0.6, min(1.4, scale))
            ss_eff = float(base) * scale
            return float(max(0.05, min(0.35, ss_eff)))
        except Exception:
            return float(max(0.05, min(0.35, base)))

    def _bounded_linear(self, value: Optional[float], center: float, span: float, min_mult: float = 0.85, max_mult: float = 1.15) -> float:
        """将指标按中心与跨度线性映射为乘子，并限制到[min,max]。value=None→1.0"""
        if value is None:
            return 1.0
        try:
            delta = (float(value) - center) / max(span, 1e-6)
            mult = 1.0 + 0.15 * delta
            return max(min_mult, min(max_mult, mult))
        except Exception:
            return 1.0

    def _per_match_rate(self, total: Optional[float], games_played: int) -> Optional[float]:
        """总量转每场速率（含平滑）。None→None"""
        if total is None:
            return None
        gp = max(games_played, 1)
        try:
            return float(total) / gp
        except Exception:
            return None

    def normalize_stats(self, team: TeamStats) -> Tuple[float, float]:
        """用非点球xG/xGA优先作为赛季基准，回退到xG/xGA；并进行轻微平滑。"""
        gp = max(team.games_played, 1)
        base_xg_total = team.npxg if isinstance(team.npxg, (int, float)) else team.xg
        base_xga_total = team.npxga if isinstance(team.npxga, (int, float)) else team.xga
        avg_xg = (base_xg_total / gp) * (1.0 + self.strength_influence_factor * 0.1)
        avg_xga = (base_xga_total / gp) * (1.0 + self.strength_influence_factor * 0.1)
        return avg_xg, avg_xga

    def calculate_rss(self, ranking: int) -> float:
        """根据排名计算长期实力（排名越靠前，RSS越高）"""
        r = max(min(ranking, self.total_teams), 1)
        # 简单的反比例映射，并平滑到合理范围
        rss = (self.total_teams + 1 - r) / self.avg_base_score
        return max(0.5, min(1.5, rss))

    def calculate_team_strengths(self, team: TeamStats, ranking: int) -> Tuple[float, float]:
        """计算球队攻防强度，融合 npxg/npxga、ppda/oppda、dc/odc（缺失则不影响）。"""
        avg_xg, avg_xga = self.normalize_stats(team)
        gp = max(team.games_played, 1)

        alpha = 0.6
        season_att = self._per_match_rate(getattr(team, "goals_for", None), gp)
        season_def = self._per_match_rate(getattr(team, "goals_against", None), gp)
        recent_att = (team.recent_goals_scored / 6.0) if isinstance(team.recent_goals_scored, (int, float)) else None
        recent_def = (team.recent_goals_conceded / 6.0) if isinstance(team.recent_goals_conceded, (int, float)) else None
        primary_att = season_att if season_att is not None else (recent_att if recent_att is not None else avg_xg)
        primary_def = season_def if season_def is not None else (recent_def if recent_def is not None else avg_xga)
        short_term_attack = (alpha * primary_att) + ((1 - alpha) * avg_xg)
        short_term_defense = (alpha * primary_def) + ((1 - alpha) * avg_xga)

        # 指标转速率
        dc_rate = self._per_match_rate(team.dc, gp)
        odc_rate = self._per_match_rate(team.odc, gp)

        # 乘子：进攻受 oppda（对手压迫强度）与自方 dc（危险机会）影响
        attack_mult = (
            self._bounded_linear(team.oppda, center=10.0, span=6.0) *
            self._bounded_linear(dc_rate, center=6.0, span=4.0)
        )
        # 乘子：防守受 ppda（自方压迫强度，越低越强）与 odc（被创造危险机会）影响
        defense_mult = (
            1.0 / self._bounded_linear(team.ppda, center=10.0, span=6.0) *
            1.0 / self._bounded_linear(odc_rate, center=6.0, span=4.0)
        )

        rss = self.calculate_rss(ranking)

        attack_strength = short_term_attack * attack_mult * (1 - self.ranking_weight) + rss * self.ranking_weight
        defense_strength = short_term_defense * defense_mult * (1 - self.ranking_weight) + (1 / rss) * self.ranking_weight
        
        return attack_strength, defense_strength

    def _adaptive_goal_cap(self, mu: float) -> int:
        """给定泊松均值mu，选择使CDF>=tail_mass的最小整数上限，夹在[min,max]。"""
        g = self.cap_min
        while g < self.cap_max and poisson.cdf(g, mu) < self.tail_mass:
            g += 1
        return max(self.cap_min, min(self.cap_max, g))

    def calculate_match_probabilities(self, home_team: TeamStats, away_team: TeamStats, overrides: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], List[ScoreProbability], np.ndarray]:
        """计算比赛结果概率，并返回比分概率矩阵（自适应维度）。
        支持外部覆盖：mu_home/mu_away（直接指定泊松/NB均值），use_nb/nb_k（负二项），use_zip/zip_pi_home/zip_pi_away（零膨胀）。
        """
        overrides = overrides or {}
        # 外部 mu 覆盖（若提供则跳过内生强度计算）
        mu_home = overrides.get("mu_home")
        mu_away = overrides.get("mu_away")
        if mu_home is None or mu_away is None:
            # 计算双方攻防强度
            home_attack, home_defense = self.calculate_team_strengths(home_team, home_team.ranking)
            away_attack, away_defense = self.calculate_team_strengths(away_team, away_team.ranking)
            # 预期进球（联赛基线、节奏与主场优势）
            base = float(overrides.get("league_base_rate", getattr(settings, "league_base_rate", 1.0)))
            tempo = float(overrides.get("league_tempo", getattr(settings, "league_tempo", 1.0)))
            ha = float(overrides.get("home_advantage", getattr(settings, "home_advantage", 1.0)))
            expected_home_goals = base * tempo * ha * home_attack * away_defense
            expected_away_goals = base * tempo * away_attack * home_defense
            mu_home, mu_away = expected_home_goals, expected_away_goals
        
        # 自适应上限（分别计算后取max，保证方阵）
        cap_h = self._adaptive_goal_cap(expected_home_goals)
        cap_a = self._adaptive_goal_cap(expected_away_goals)
        self.max_goals = max(cap_h, cap_a)
        
        # 创建进球概率矩阵（维度：(max_goals+1)^2）—— 独立泊松或双变量泊松
        use_bp = bool(getattr(settings, "enable_bivariate_poisson", False))
        use_nb = bool(overrides.get("use_nb", False))
        nb_k = float(overrides.get("nb_k", 10.0))  # 形状参数（越大越近泊松）
        use_zip = bool(overrides.get("use_zip", False))
        zip_pi_home = float(overrides.get("zip_pi_home", 0.0))
        zip_pi_away = float(overrides.get("zip_pi_away", 0.0))
        if use_bp:
            ss_base = float((overrides or {}).get("bp_shared_strength", getattr(settings, "bp_shared_strength", 0.15)))
            ss = ss_base
            if getattr(settings, "enable_bp_dynamic", True):
                ss = self._dynamic_shared_strength(mu_home, mu_away, ss_base)
            prob_matrix = self._bivariate_poisson_matrix(mu_home, mu_away, ss)
        else:
            prob_matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
            # 选择分布：泊松 / 负二项 / ZIP-泊松
            if not use_nb and not use_zip:
                for i in range(self.max_goals + 1):
                    for j in range(self.max_goals + 1):
                        prob_matrix[i, j] = (poisson.pmf(i, mu_home) * poisson.pmf(j, mu_away))
            elif use_nb:
                from scipy.stats import nbinom
                # 参数化：方差 = mu + mu^2/k → p = k/(k+mu)
                k_h = max(1e-6, nb_k)
                k_a = max(1e-6, nb_k)
                p_h = k_h / (k_h + max(1e-9, mu_home))
                p_a = k_a / (k_a + max(1e-9, mu_away))
                for i in range(self.max_goals + 1):
                    for j in range(self.max_goals + 1):
                        prob_matrix[i, j] = (nbinom.pmf(i, k_h, p_h) * nbinom.pmf(j, k_a, p_a))
            else:  # use_zip
                # 零膨胀泊松：P(X=0)=pi+(1-pi)*Poisson(0;mu)，P(X>0)=(1-pi)*Poisson(k;mu)
                for i in range(self.max_goals + 1):
                    p_i = (zip_pi_home + (1.0 - zip_pi_home) * poisson.pmf(0, mu_home)) if i == 0 else ((1.0 - zip_pi_home) * poisson.pmf(i, mu_home))
                    for j in range(self.max_goals + 1):
                        p_j = (zip_pi_away + (1.0 - zip_pi_away) * poisson.pmf(0, mu_away)) if j == 0 else ((1.0 - zip_pi_away) * poisson.pmf(j, mu_away))
                        prob_matrix[i, j] = p_i * p_j

        # 应用 DC 相关性与平局加强（可开关）
        try:
            # 双变量泊松已隐含相关性，默认跳过 DC 修正；支持 overrides 覆盖
            use_dc = bool(overrides.get("use_dc", getattr(settings, "enable_dc_correlation", False)))
            if (not use_bp) and use_dc:
                rho = float(overrides.get("dc_rho", getattr(settings, "dc_rho", -0.12)))
                prob_matrix = self._apply_dc_correlation(prob_matrix, mu_home, mu_away, rho)
            if getattr(settings, "enable_draw_boost", False):
                strength = float((overrides or {}).get("draw_boost_strength", getattr(settings, "draw_boost_strength", 0.08)))
                prob_matrix = self._apply_draw_boost(prob_matrix, mu_home, mu_away, strength)
        except Exception:
            prob_matrix = self._renorm_matrix(prob_matrix)
        
        # 计算胜平负概率
        home_win_prob = float(np.sum(np.tril(prob_matrix, -1)))
        draw_prob = float(np.sum(np.diag(prob_matrix)))
        away_win_prob = float(np.sum(np.triu(prob_matrix, 1)))
        
        # 标准化概率（确保三类合计1）
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        
        # 获取最可能的比分（取前三）
        score_probs = []
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                score_probs.append(ScoreProbability(
                    score=f"{i}-{j}",
                    probability=float(prob_matrix[i, j])
                ))
        score_probs.sort(key=lambda x: x.probability, reverse=True)
        score_probs = score_probs[:3]
        
        return {
            "home_win": home_win_prob,
            "draw": draw_prob,
            "away_win": away_win_prob
        }, score_probs, prob_matrix

    def expected_goal_diff(self, home_team: TeamStats, away_team: TeamStats, overrides: Optional[Dict[str, float]] = None) -> float:
        """返回基于模型的预期净胜球（主队-客队）。"""
        home_attack, home_defense = self.calculate_team_strengths(home_team, home_team.ranking)
        away_attack, away_defense = self.calculate_team_strengths(away_team, away_team.ranking)
        base = float((overrides or {}).get("league_base_rate", getattr(settings, "league_base_rate", 1.0)))
        tempo = float((overrides or {}).get("league_tempo", getattr(settings, "league_tempo", 1.0)))
        ha = float((overrides or {}).get("home_advantage", getattr(settings, "home_advantage", 1.0)))
        expected_home_goals = base * tempo * ha * home_attack * away_defense
        expected_away_goals = base * tempo * away_attack * home_defense
        return float(expected_home_goals - expected_away_goals)