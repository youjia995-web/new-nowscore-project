from typing import Dict, List, Tuple, Optional
from ..models.match import OddsData
from ..services.company_weight import CompanyWeightService
from ..config import settings
import numpy as np

class MarketEngine:
    def __init__(self):
        pass
        
    def _safe_implied(self, home: float, draw: float, away: float) -> Dict[str, float]:
        """将三方赔率转换为隐含概率，并做去水归一；对缺失或异常值做兜底。"""
        vals = [home, draw, away]
        if any(v is None or v <= 0 for v in vals):
            return {"home_win": None, "draw": None, "away_win": None}
        margin = (1/home + 1/draw + 1/away)
        if margin <= 0:
            return {"home_win": None, "draw": None, "away_win": None}
        return {
            "home_win": (1/home) / margin,
            "draw": (1/draw) / margin,
            "away_win": (1/away) / margin,
        }
        
    def calculate_implied_probabilities(self, odds: OddsData) -> Tuple[Dict[str, float], str]:
        """计算赔率隐含概率并分析市场趋势（优先使用终盘，缺失时回退初盘）。"""
        # 终盘优先；若有任一终盘缺失或非法，则回退初盘
        probs_final = self._safe_implied(odds.final_home_win, odds.final_draw, odds.final_away_win)
        use_initial = any(probs_final[k] is None for k in ["home_win", "draw", "away_win"])
        if use_initial:
            probs_initial = self._safe_implied(odds.initial_home_win, odds.initial_draw, odds.initial_away_win)
            probs = probs_initial
        else:
            probs = probs_final
        
        # 分析市场趋势（比较初终盘概率与盘口变化）
        initial_probs = self._safe_implied(odds.initial_home_win, odds.initial_draw, odds.initial_away_win)
        final_probs = probs_final
        trend = self._analyze_trend(
            initial_probs=(initial_probs.get("home_win") or 0.0, initial_probs.get("draw") or 0.0, initial_probs.get("away_win") or 0.0),
            final_probs=(final_probs.get("home_win") or 0.0, final_probs.get("draw") or 0.0, final_probs.get("away_win") or 0.0),
            initial_handicap=odds.initial_handicap,
            final_handicap=odds.final_handicap
        )
        
        return probs, trend
    
    def _company_weight(self, company: str) -> float:
        """公司权重（历史稳定性/成交量启发式）。未识别公司使用均匀权重。"""
        weights = {
            "365": 0.30,
            "皇冠（crown）": 0.30,
            "澳门": 0.20,
            "易胜博": 0.20,
        }
        return weights.get(company, 0.25)
    
    def analyze_market_consensus(self, odds_data: List[OddsData]) -> Tuple[Dict[str, float], str]:
        """分析多家公司赔率，得出市场共识（加权归一）。"""
        if not odds_data:
            return {"home_win": None, "draw": None, "away_win": None}, "无可用赔率数据"
        weighted_home = weighted_draw = weighted_away = 0.0
        weight_sum = 0.0
        all_trends: List[str] = []
        # 接入学习型公司权重
        cw = CompanyWeightService(settings.db_path)
        # 将 OddsData 转为简易字典以进行学习
        cw_rows: List[dict] = []
        for odds in odds_data:
            cw_rows.append({
                "company": getattr(odds, "company", None),
                "initial_home_odds": getattr(odds, "initial_home_win", None),
                "initial_draw_odds": getattr(odds, "initial_draw", None),
                "initial_away_odds": getattr(odds, "initial_away_win", None),
                "final_home_odds": getattr(odds, "final_home_win", None),
                "final_draw_odds": getattr(odds, "final_draw", None),
                "final_away_odds": getattr(odds, "final_away_win", None),
            })
        cw.learn_from_odds(cw_rows)
        # 批量取权重
        companies = [getattr(o, "company", None) or "Unknown" for o in odds_data]
        learnt_weights = cw.batch_get(companies)

        for odds in odds_data:
            probs, trend = self.calculate_implied_probabilities(odds)
            if any(probs[k] is None for k in ["home_win", "draw", "away_win"]):
                continue
            company = getattr(odds, "company", None) or "Unknown"
            w = learnt_weights.get(company, self._company_weight(company))
            weighted_home += probs["home_win"] * w
            weighted_draw += probs["draw"] * w
            weighted_away += probs["away_win"] * w
            weight_sum += w
            all_trends.append(trend)
        if weight_sum <= 0:
            return {"home_win": None, "draw": None, "away_win": None}, "赔率数据异常"
        avg_home = weighted_home / weight_sum
        avg_draw = weighted_draw / weight_sum
        avg_away = weighted_away / weight_sum
        
        # 综合市场趋势
        final_trend = self._combine_trends(all_trends)

        # 聚合盘口趋势并添加到趋势描述
        handicap_summary = self._aggregate_handicap_trend(odds_data)
        if handicap_summary:
            final_trend = f"{final_trend}；{handicap_summary}"
        
        return {
            "home_win": avg_home,
            "draw": avg_draw,
            "away_win": avg_away
        }, final_trend

    def _analyze_trend(self, initial_probs: Tuple[float, float, float],
                      final_probs: Tuple[float, float, float],
                      initial_handicap: str,
                      final_handicap: str) -> str:
        """分析单家公司的赔率变化趋势"""
        initial_home, initial_draw, initial_away = initial_probs
        final_home, final_draw, final_away = final_probs
        
        trends = []
        
        # 分析胜平负概率变化
        home_change = final_home - initial_home
        draw_change = final_draw - initial_draw
        away_change = final_away - initial_away
        
        if abs(home_change) > 0.02:  # 2%的变化阈值
            trends.append("主胜赔率" + ("下调" if home_change > 0 else "上调"))
        if abs(away_change) > 0.02:
            trends.append("客胜赔率" + ("下调" if away_change > 0 else "上调"))
            
        # 分析盘口变化
        if initial_handicap != final_handicap:
            trends.append(f"盘口从{initial_handicap}变为{final_handicap}")
            
        if not trends:
            return "市场保持稳定"
        return "，".join(trends)

    def _combine_trends(self, trends: List[str]) -> str:
        """综合多家公司的趋势分析"""
        if all(trend == "市场保持稳定" for trend in trends):
            return "市场整体保持稳定"
            
        # 统计趋势关键词
        trend_keywords = {
            "主胜赔率下调": 0,
            "主胜赔率上调": 0,
            "客胜赔率下调": 0,
            "客胜赔率上调": 0
        }
        
        for trend in trends:
            for keyword in trend_keywords:
                if keyword in trend:
                    trend_keywords[keyword] += 1
                    
        # 生成综合趋势描述
        final_trends = []
        if trend_keywords["主胜赔率下调"] > trend_keywords["主胜赔率上调"]:
            final_trends.append("市场资金整体看好主胜")
        elif trend_keywords["主胜赔率上调"] > trend_keywords["主胜赔率下调"]:
            final_trends.append("市场资金整体看空主胜")
            
        if trend_keywords["客胜赔率下调"] > trend_keywords["客胜赔率上调"]:
            final_trends.append("市场资金整体看好客胜")
        elif trend_keywords["客胜赔率上调"] > trend_keywords["客胜赔率下调"]:
            final_trends.append("市场资金整体看空客胜")
            
        return "，".join(final_trends) if final_trends else "市场观点分歧"

    def _extract_handicap_numbers(self, val: str) -> List[float]:
        """从让球字符串中提取所有数值，支持常见中文写法。
        例如 "两球半(2.5)"、"2.5/3"、"半一"、"平/半" → 返回数值列表。
        """
        if val is None:
            return []
        s = str(val).strip()
        import re
        nums = re.findall(r"[+-]?\d+(?:\.\d+)?", s)
        if nums:
            try:
                return [float(x) for x in nums]
            except Exception:
                return []
        # 无显式数字时，识别中文盘口词
        token_map = {
            "平手": [0.0], "pk": [0.0],
            "平/半": [0.0, 0.5], "平半": [0.0, 0.5],
            "半球": [0.5],
            "半/一": [0.5, 1.0], "半一": [0.5, 1.0],
            "一球": [1.0],
            "一/球半": [1.0, 1.5], "一球/球半": [1.0, 1.5], "一球半": [1.5], "球半": [1.5],
            "两球": [2.0],
            "两球/两球半": [2.0, 2.5], "两球半": [2.5],
            "两球半/三球": [2.5, 3.0], "三球": [3.0],
            "三球/三球半": [3.0, 3.5], "三球半": [3.5],
        }
        s_norm = s.replace("／", "/").replace(" ", "")
        for k, v in token_map.items():
            if k in s_norm:
                return v
        return []

    def _parse_handicap_struct(self, val: str) -> Tuple[Optional[float], Optional[bool]]:
        """解析让球字符串的数值与方向。
        返回 (line, home_favored)。line>0 表示主让，<0 表示主受让；None 表示无法解析。
        识别关键词："受"、"主让"、"客让"，以及显式正负号与中文写法。
        """
        if not val:
            return None, None
        s = str(val)
        nums = self._extract_handicap_numbers(s)
        if not nums:
            return None, None
        avg = sum(nums) / len(nums)
        # 方向识别（与显示约定一致：+ 主受让，- 主让）
        s_norm = s.lower()
        home_favored = None
        if ("受" in s) or ("主受让" in s) or ("客让" in s):
            home_favored = False
        elif ("主让" in s):
            home_favored = True
        elif s.strip().startswith("+"):
            home_favored = False
        elif s.strip().startswith("-"):
            home_favored = True
        # 根据方向调整符号（统一内部：正=主让，负=主受让）
        if home_favored is None:
            line = avg
        else:
            mag = abs(avg)
            line = mag if home_favored else -mag
        return line, home_favored

    def _aggregate_handicap_trend(self, odds_data: List[OddsData]) -> str:
        """统计各公司让球从初盘到终盘的方向与力度，输出聚合结论。"""
        if not odds_data:
            return ""
        diffs = []
        init_vals = []
        final_vals = []
        up = down = flat = 0
        for o in odds_data:
            init_line, _ = self._parse_handicap_struct(o.initial_handicap)
            final_line, _ = self._parse_handicap_struct(o.final_handicap)
            if init_line is None or final_line is None:
                continue
            diff = final_line - init_line
            diffs.append(diff)
            init_vals.append(init_line)
            final_vals.append(final_line)
            if diff > 1e-6:
                up += 1
            elif diff < -1e-6:
                down += 1
            else:
                flat += 1
        if not diffs:
            return "盘口数据不足，无法聚合趋势"
        avg_init = sum(init_vals) / len(init_vals)
        avg_final = sum(final_vals) / len(final_vals)
        avg_diff = sum(diffs) / len(diffs)
        majority = "升盘" if up > max(down, flat) else ("降盘" if down > max(up, flat) else "盘口稳定")
        dir = "升盘（主让加深）" if avg_diff > 0 else ("降盘（主让减弱）" if avg_diff < 0 else "盘口保持不变")
        def fmt(n: float) -> str:
            return f"{n:+.2f}" if abs(n) < 10 else f"{n:.2f}"
        return (
            f"盘口聚合：平均初盘{avg_init:.2f} → 终盘{avg_final:.2f}，均值变化{fmt(avg_diff)}；"
            f"公司方向：升盘{up}家、降盘{down}家、不变{flat}家（多数{majority}，方向：{dir}）"
        )

    # 新增：异常阈值配置与标定骨架
    def _get_anomaly_thresholds(self) -> Dict[str, float]:
        """返回用于标签判定与评分缩放的阈值。
        若 settings.enable_hcap_quantile_calibration 为 True，可在后续接入基于历史库的分位标定。
        当前骨架：使用配置或默认值。
        """
        minor = getattr(settings, "hcap_threshold_minor", 0.25)
        major = getattr(settings, "hcap_threshold_major", 0.50)
        full = getattr(settings, "hcap_full_score_delta", 0.50)
        return {"minor": float(minor), "major": float(major), "full": float(full)}

    def compute_fair_handicap(self, prob_matrix) -> float:
        """基于比分概率矩阵计算公平让球线（四分之一盘刻度）。
        目标：在均水条件下，使主队亚洲盘的期望收益≈0。
        候选范围：[-3.0, 3.0]，步长0.25。
        """
        try:
            pm = np.array(prob_matrix, dtype=float)
            hdim, wdim = pm.shape
            # 构建净胜球分布 D = i - j
            diff_probs: Dict[int, float] = {}
            for i in range(hdim):
                for j in range(wdim):
                    k = i - j
                    diff_probs[k] = diff_probs.get(k, 0.0) + float(pm[i, j])
            # 归一化以防数值误差
            total = sum(diff_probs.values())
            if total > 0:
                for k in list(diff_probs.keys()):
                    diff_probs[k] /= total

            def p_gt(x: float) -> float:
                return sum(p for k, p in diff_probs.items() if k > x)
            def p_lt(x: float) -> float:
                return sum(p for k, p in diff_probs.items() if k < x)

            def ev_int(H: float) -> float:
                # 整盘：赢 P(D>H)，输 P(D<H)，走水 P(D==H)
                return p_gt(H) - p_lt(H)
            def ev_half(H: float) -> float:
                # 半盘：无走水，赢 P(D>H)，输 P(D<H)
                return p_gt(H) - p_lt(H)
            def ev_quarter(H: float) -> float:
                base = np.floor(H)
                frac = H - base
                eps = 1e-9
                if abs(frac - 0.0) < eps:
                    return ev_int(H)
                if abs(frac - 0.50) < eps:
                    return ev_half(H)
                if abs(frac - 0.25) < eps:
                    return 0.5 * ev_int(base) + 0.5 * ev_half(base + 0.50)
                if abs(frac - 0.75) < eps:
                    return 0.5 * ev_half(base + 0.50) + 0.5 * ev_int(base + 1.0)
                # 非标准刻度，回退为四舍五入到最近四分之一盘
                Hq = round(H * 4) / 4.0
                return ev_quarter(Hq)

            # 候选集合（根据矩阵维度限制极端范围）
            max_diff = max(abs(k) for k in diff_probs.keys()) if diff_probs else 3
            span = float(min(3.0, max_diff + 0.5))
            candidates = [round(x * 4) / 4.0 for x in np.arange(-span, span + 0.001, 0.25)]
            # 找到 |EV| 最小的线；并偏好绝对值更小的线
            best = None
            best_key = (float('inf'), float('inf'))  # (abs(EV), abs(H))
            for H in candidates:
                ev = ev_quarter(H)
                key = (abs(ev), abs(H))
                if key < best_key:
                    best_key = key
                    best = H
            return float(best if best is not None else 0.0)
        except Exception:
            # 失败时回退零盘
            return 0.0

    # —— 让盘合理性与异象评估 ——
    def expected_handicap_from_goal_diff(self, goal_diff: float) -> float:
        """将模型的预期净胜球映射为常见让球线（四分之一盘刻度）。支持标定回退。"""
        try:
            from ..services.calibration import CalibrationService
            from ..config import settings as _settings
            cs = CalibrationService(_settings.db_path)
            return cs.map_goal_diff_to_handicap(goal_diff)
        except Exception:
            x = abs(goal_diff)
            if x < 0.10:
                base = 0.0
            elif x < 0.30:
                base = 0.25
            elif x < 0.50:
                base = 0.50
            elif x < 0.80:
                base = 0.75
            elif x < 1.10:
                base = 1.00
            elif x < 1.40:
                base = 1.25
            elif x < 1.70:
                base = 1.50
            elif x < 2.00:
                base = 1.75
            else:
                base = 2.00
            return base if goal_diff >= 0 else -base

    def market_stability_score(self, odds_data) -> float:
        """基于多家公司终盘隐含概率的方差估计市场稳定性，返回[0,1]。"""
        try:
            import numpy as np
        except Exception:
            # 若 numpy 不可用，返回中性值
            return 0.5
        triples = []
        for o in (odds_data or []):
            probs = self._safe_implied(getattr(o, "final_home_win", None), getattr(o, "final_draw", None), getattr(o, "final_away_win", None))
            if probs and all(k in probs and isinstance(probs[k], (int, float)) for k in ("home_win", "draw", "away_win")):
                triples.append([float(probs["home_win"]), float(probs["draw"]), float(probs["away_win"])])
        if len(triples) < 2:
            return 0.5
        arr = np.array(triples, dtype=float)
        var = float(np.var(arr, axis=0).mean())
        alpha = 15.0  # 映射强度：方差越小，稳定性越高
        score = 1.0 / (1.0 + alpha * var)
        return float(max(0.0, min(1.0, score)))

    def assess_handicap_anomaly(self, expected_handicap: float, odds_data: List[OddsData]) -> Dict[str, float]:
        """
        评估让盘是否合理：
        - 聚合多家公司终盘让球为一个均值（识别“受/主让/客让”方向）
        - 计算与模型期望让球的偏差
        - 根据水位（主/客）给出偏向说明，并输出水位差
        - 使用水位隐含两向概率对异常分数进行修正（连续、方向对称）
        返回字典包含：market_final_handicap, handicap_anomaly_score, handicap_anomaly_label, handicap_water_bias, handicap_water_diff
        """
        finals = []
        initials = []
        home_waters = []
        away_waters = []
        for o in odds_data:
            f_line, _ = self._parse_handicap_struct(o.final_handicap)
            i_line, _ = self._parse_handicap_struct(o.initial_handicap)
            if f_line is not None:
                finals.append(f_line)
            if i_line is not None:
                initials.append(i_line)
            try:
                if o.final_handicap_home_odds and o.final_handicap_home_odds > 0:
                    home_waters.append(float(o.final_handicap_home_odds))
                if o.final_handicap_away_odds and o.final_handicap_away_odds > 0:
                    away_waters.append(float(o.final_handicap_away_odds))
            except Exception:
                pass
        market_final = sum(finals) / len(finals) if finals else None
        market_initial = sum(initials) / len(initials) if initials else None
        home_water = sum(home_waters) / len(home_waters) if home_waters else None
        away_water = sum(away_waters) / len(away_waters) if away_waters else None
        water_diff = None
        if home_water is not None and away_water is not None:
            water_diff = round(home_water - away_water, 4)  # 正数表示主水更高

        # 水位偏向（文本标签）
        water_bias = None
        if home_water is not None and away_water is not None:
            if home_water < 1.85 and away_water > 2.05:
                water_bias = "主赔低水（强挺主让）"
            elif home_water > 2.05 and away_water < 1.85:
                water_bias = "主赔高水（弱化主让）"
            else:
                water_bias = "水位中性"

        # 数据不足
        if market_final is None or expected_handicap is None:
            return {
                "market_final_handicap": market_final,
                "handicap_anomaly_score": None,
                "handicap_anomaly_label": "盘口数据不足",
                "handicap_water_bias": water_bias,
                "handicap_water_diff": water_diff,
                "expected_handicap": expected_handicap,
                "market_initial_handicap": market_initial,
            }

        delta = market_final - expected_handicap
        th = self._get_anomaly_thresholds()
        full_scale = th.get("full", 0.50) or 0.50
        base_score = min(1.0, abs(delta) / float(full_scale))

        # 连续的水位概率修正：依据两向赔率的隐含概率与盘口方向/偏差方向一致性
        water_adj = 0.0
        implied_home = implied_away = None
        try:
            if (home_water and home_water > 0) and (away_water and away_water > 0):
                inv_h = 1.0 / float(home_water)
                inv_a = 1.0 / float(away_water)
                margin = inv_h + inv_a
                if margin > 0:
                    implied_home = inv_h / margin
                    implied_away = inv_a / margin
                    lean = (implied_home - implied_away)  # ∈[-1,1]
                    sign_line = 1.0 if market_final >= 0 else -1.0
                    sign_delta = 1.0 if delta >= 0 else -1.0
                    strength = getattr(settings, "hcap_water_adj_strength", 0.20)
                    water_adj = float(strength) * lean * sign_line * sign_delta
        except Exception:
            pass
        anomaly_score = max(0.0, min(1.0, base_score + water_adj))

        # 标签判定（考虑方向反转，使用可配置阈值）
        minor = float(th.get("minor", 0.25))
        # 方向反转阈值（小正负窗口避免浮点误判）
        rev_eps = getattr(settings, "hcap_reverse_dir_margin", 0.10)
        label = "合理让盘" if abs(delta) <= minor else ""
        if not label:
            if expected_handicap >= 0 and market_final < -float(rev_eps):
                label = "盘面方向与模型相反（客让）"
            elif expected_handicap <= 0 and market_final > float(rev_eps):
                label = "盘面方向与模型相反（主让）"
            else:
                if delta > minor:
                    label = "让盘偏深"
                elif delta < -minor:
                    label = "让盘偏浅"
                else:
                    label = "轻微偏差（近合理）"

        return {
            "market_final_handicap": market_final,
            "handicap_anomaly_score": anomaly_score,
            "handicap_anomaly_label": label,
            "handicap_water_bias": water_bias,
            "handicap_water_diff": water_diff,
            "expected_handicap": expected_handicap,
            "market_initial_handicap": market_initial,
        }