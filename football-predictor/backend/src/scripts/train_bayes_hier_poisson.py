"""
使用 PyMC 训练分层贝叶斯泊松/负二项模型：
- 团队层：攻 att_t、守 def_t（均值约束为 0）
- 联赛/全局：log_base、home_advantage_log、tempo_log（可选）
- 观测：home_goals, away_goals
- 可选：负二项超参数 nb_k（过度离散），零膨胀 zip_pi_home/zip_pi_away（加强 0）

训练完成后，将参数写入 calibration_params（key='bayes_hier_params_v1'）。
"""

import argparse
import numpy as np
import pymc as pm
import arviz as az
from typing import Dict, Any

from ..services.bayes_params_store import BayesParamsStore
from ..services.calibration import CalibrationService
from ..data.prediction_store import PredictionStore


def load_matches(days: int = 3650) -> Dict[str, Any]:
    store = PredictionStore()
    # 这里假设提供获取历史真值的接口：返回包含 home_team, away_team, home_goals, away_goals, league, match_date
    # 若项目中不存在统一方法，可替代为从 predictions 表读取有实际结果的记录。
    df = store.fetch_historical_results(limit_days=days)
    teams = sorted(set(df['home_team']).union(set(df['away_team'])))
    team_index = {t: i for i, t in enumerate(teams)}
    home_idx = np.array([team_index[t] for t in df['home_team']], dtype=int)
    away_idx = np.array([team_index[t] for t in df['away_team']], dtype=int)
    home_goals = np.array(df['home_goals'], dtype=int)
    away_goals = np.array(df['away_goals'], dtype=int)
    league = np.array(df.get('league', ['']*len(df)), dtype=object)
    # 时间衰减权重
    days_ago = np.array(df.get('days_ago', [0]*len(df)), dtype=float)
    return {
        'teams': teams,
        'home_idx': home_idx,
        'away_idx': away_idx,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'league': league,
        'days_ago': days_ago,
    }


def fit_model(data: Dict[str, Any], use_nb: bool, decay: float) -> Dict[str, Any]:
    n_team = len(data['teams'])
    h = data['home_idx']
    a = data['away_idx']
    y_h = data['home_goals']
    y_a = data['away_goals']
    w = np.exp(-decay * data['days_ago']).astype(float)
    w = np.clip(w, 1e-3, 1.0)

    with pm.Model() as m:
        # 团队层攻防参数（均值约束）
        att = pm.Normal('att_raw', 0.0, 1.0, shape=n_team)
        deff = pm.Normal('def_raw', 0.0, 1.0, shape=n_team)
        att_c = pm.Deterministic('att', att - pm.math.mean(att))
        def_c = pm.Deterministic('def', deff - pm.math.mean(deff))

        # 全局参数（log 尺度）
        log_base = pm.Normal('log_base', np.log(1.4), 0.5)
        ha_log = pm.Normal('home_advantage_log', np.log(1.10), 0.3)
        tempo_log = pm.Normal('tempo_log', 0.0, 0.3)

        mu_h_log = log_base + tempo_log + ha_log + att_c[h] - def_c[a]
        mu_a_log = log_base + tempo_log + att_c[a] - def_c[h]
        mu_h = pm.Deterministic('mu_home', pm.math.exp(mu_h_log))
        mu_a = pm.Deterministic('mu_away', pm.math.exp(mu_a_log))

        if use_nb:
            # 负二项形状参数（越大越接近泊松）
            k = pm.HalfNormal('nb_k', 5.0)
            # 以权重拟合：通过自定义对数似然加权
            def nb_ll(y, mu, k, w):
                # p = k/(k+mu), pm.NegativeBinomial 参数化为 mu, alpha = 1/k
                return pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=1.0/pm.math.maximum(k, 1e-6)), y) * w
            pm.Potential('ll_home', nb_ll(y_h, mu_h, k, w))
            pm.Potential('ll_away', nb_ll(y_a, mu_a, k, w))
        else:
            pm.Potential('ll_home', pm.logp(pm.Poisson.dist(mu=mu_h), y_h) * w)
            pm.Potential('ll_away', pm.logp(pm.Poisson.dist(mu=mu_a), y_a) * w)

        idata = pm.sample(1000, tune=1000, target_accept=0.9, chains=2, cores=2)

    # 后验均值作为点估计
    post = az.extract(idata, num_samples=None)
    att_mean = post['att'].mean(axis=0)
    def_mean = post['def'].mean(axis=0)
    result = {
        'team_attack': {t: float(att_mean[i]) for i, t in enumerate(data['teams'])},
        'team_defense': {t: float(def_mean[i]) for i, t in enumerate(data['teams'])},
        'log_base': float(post['log_base'].mean()),
        'home_advantage_log': float(post['home_advantage_log'].mean()),
        'tempo_log': float(post['tempo_log'].mean()),
    }
    if use_nb:
        result['nb_k'] = float(post['nb_k'].mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=3650)
    parser.add_argument('--decay', type=float, default=0.002, help='时间衰减系数（每天）')
    parser.add_argument('--use-nb', action='store_true', help='使用负二项以处理过度离散')
    args = parser.parse_args()

    data = load_matches(days=args.days)
    params = fit_model(data, use_nb=args.use_nb, decay=args.decay)

    # 保存参数
    store = BayesParamsStore()
    store.save(params)
    print('Saved bayes params to key:', store.DEFAULT_KEY)


if __name__ == '__main__':
    main()