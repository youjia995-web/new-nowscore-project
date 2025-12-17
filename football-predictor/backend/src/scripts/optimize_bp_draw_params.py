"""
对已训练的贝叶斯团队参数进行最大似然网格搜索：
- bp_shared_strength（双变量泊松共享强度）
- dc_rho（Dixon-Coles 相关系数，若开启）
- draw_boost_strength（平局加强）
遍历网格后选择使比赛三路结果对数似然最大的组合，并写回参数存储。
"""

import argparse
import numpy as np
from typing import Dict, Any, Tuple

from ..services.bayes_params_store import BayesParamsStore
from ..data.prediction_store import PredictionStore
from ..engines.poisson_engine import PoissonEngine


def outcome_to_idx(hg: int, ag: int) -> int:
    if hg > ag:
        return 0  # H
    elif hg == ag:
        return 1  # D
    else:
        return 2  # A


def log_likelihood_for_grid(params: Dict[str, Any], matches, ss: float, rho: float, draw_boost: float) -> float:
    pe = PoissonEngine()
    ll = 0.0
    for m in matches:
        # 计算 mu_home, mu_away
        att_h = float(params['team_attack'].get(m['home_team'], 0.0))
        def_h = float(params['team_defense'].get(m['home_team'], 0.0))
        att_a = float(params['team_attack'].get(m['away_team'], 0.0))
        def_a = float(params['team_defense'].get(m['away_team'], 0.0))
        mu_h = float(np.exp(params['log_base'] + params['tempo_log'] + params['home_advantage_log'] + att_h - def_a))
        mu_a = float(np.exp(params['log_base'] + params['tempo_log'] + att_a - def_h))

        overrides = {
            'mu_home': mu_h,
            'mu_away': mu_a,
            'bp_shared_strength': ss,
            'draw_boost_strength': draw_boost,
            'use_dc': True,
            'dc_rho': rho,
        }

        probs, _, _ = pe.calculate_match_probabilities(m['home_stats'], m['away_stats'], overrides)
        p = [probs['home_win'], probs['draw'], probs['away_win']]
        ll += np.log(max(1e-12, p[outcome_to_idx(m['home_goals'], m['away_goals'])]))
    return ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss-min', type=float, default=0.05)
    parser.add_argument('--ss-max', type=float, default=0.35)
    parser.add_argument('--ss-step', type=float, default=0.02)
    parser.add_argument('--draw-min', type=float, default=0.0)
    parser.add_argument('--draw-max', type=float, default=0.15)
    parser.add_argument('--draw-step', type=float, default=0.01)
    args = parser.parse_args()

    # 加载参数与比赛
    store = BayesParamsStore()
    params = store.load() or {}
    ps = PredictionStore()
    df = ps.fetch_historical_results(limit_days=3650)
    # 将比赛打包为所需字典列表（依赖现有 TeamStats 构造）
    matches = []
    for i in range(len(df['home_team'])):
        matches.append({
            'home_team': df['home_team'][i],
            'away_team': df['away_team'][i],
            'home_goals': int(df['home_goals'][i]),
            'away_goals': int(df['away_goals'][i]),
            'home_stats': ps.get_team_stats(df['home_team'][i]),
            'away_stats': ps.get_team_stats(df['away_team'][i]),
        })

    best = (-np.inf, None)
    for ss in np.arange(args.ss_min, args.ss_max + 1e-9, args.ss_step):
        for draw_boost in np.arange(args.draw_min, args.draw_max + 1e-9, args.draw_step):
            # 可按需扩展对 rho 的遍历，这里先使用存储或默认值
            rho = float(params.get('dc_rho', -0.12))
            ll = log_likelihood_for_grid(params, matches, ss, rho=rho, draw_boost=draw_boost)
            if ll > best[0]:
                best = (ll, (ss, draw_boost))

    (best_ss, best_draw) = best[1]
    params['bp_shared_strength'] = float(best_ss)
    params['draw_boost_strength'] = float(best_draw)
    store.save(params)
    print('Optimized bp_shared_strength=', best_ss, 'draw_boost_strength=', best_draw)


if __name__ == '__main__':
    main()