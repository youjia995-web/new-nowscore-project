import json
import requests
from typing import Dict, List
from ..models.match import TeamStats, ScoreProbability
from ..config import settings

class AIEngine:
    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.api_base = settings.deepseek_api_base
        # 无秘钥时禁用外网调用，直接使用本地fallback
        self.enabled = bool(self.api_key)
        # 记录实际分析来源：'deepseek' 或 'fallback'
        self.last_source = None

    def generate_analysis(self, 
                        home_team: TeamStats,
                        away_team: TeamStats,
                        model_probs: Dict[str, float],
                        model_scores: List[ScoreProbability],
                        market_probs: Dict[str, float],
                        market_trend: str,
                        history_summary: str = "",
                        market_history_summary: str = "",
                        market_history_stats: Dict = None,
                        fused_probs: Dict[str, float] = None,
                        fusion_weights: Dict = None,
                        fusion_notes: str = "",
                        handicap_expected: float = None,
                        handicap_market_initial: float = None,
                        handicap_market_final: float = None,
                        handicap_anomaly_label: str = "",
                        handicap_anomaly_score: float = None,
                        handicap_water_bias: str = "") -> str:
        """生成AI分析报告"""
        
        # 若未配置秘钥，视配置决定是否允许fallback
        if not self.enabled:
            if settings.require_ai_api:
                raise RuntimeError("AI API required but not configured (missing DEEPSEEK_API_KEY)")
            self.last_source = 'fallback'
            return self._generate_fallback_analysis(
                model_probs, market_probs, market_trend, fused_probs, fusion_weights, fusion_notes,
                model_scores=model_scores,
                handicap_expected=handicap_expected,
                handicap_market_initial=handicap_market_initial,
                handicap_market_final=handicap_market_final,
                handicap_anomaly_label=handicap_anomaly_label,
                handicap_anomaly_score=handicap_anomaly_score,
                handicap_water_bias=handicap_water_bias,
            )
        
        # 构建prompt
        prompt = self._build_prompt(
            home_team, away_team,
            model_probs, model_scores,
            market_probs, market_trend,
            history_summary,
            market_history_summary,
            market_history_stats,
            fused_probs,
            fusion_weights,
            fusion_notes,
            handicap_expected=handicap_expected,
            handicap_market_initial=handicap_market_initial,
            handicap_market_final=handicap_market_final,
            handicap_anomaly_label=handicap_anomaly_label,
            handicap_anomaly_score=handicap_anomaly_score,
            handicap_water_bias=handicap_water_bias,
        )
        
        try:
            # 调用DeepSeek API
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是资深足彩分析师与AI模型工程师。"
                                "任务：基于提供的数据进行专业、结构化的比赛分析。"
                                "要求：不虚构数据、不越界推断、不提供下注建议；"
                                "中文输出；用分段标题与要点列表；"
                                "在开头给出“摘要”：胜平负百分比分布（优先使用融合概率；否则使用模型概率），"
                                "以及前3个可能比分及其概率（例如 1-0(18%)，1-1(14%)，2-1(12%)）。"
                                "随后给出模型与市场差异、让盘与水位、历史维度、融合概率、风险提示与结论。"
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4096
                }
            )
            
            if response.status_code == 200:
                analysis = response.json()["choices"][0]["message"]["content"]
                # 保障完整性：若缺失“结论”等尾段，做本地补全
                analysis = self._ensure_tail(
                    analysis,
                    fused_probs,
                    fusion_weights,
                    fusion_notes,
                    handicap_expected,
                    handicap_market_initial,
                    handicap_market_final,
                    handicap_anomaly_label,
                    handicap_anomaly_score,
                    handicap_water_bias,
                )
                self.last_source = 'deepseek'
                return analysis
            else:
                if settings.require_ai_api:
                    raise RuntimeError(f"AI API required but call failed: {response.status_code} {response.text}")
                self.last_source = 'fallback'
                return self._generate_fallback_analysis(
                    model_probs, market_probs, market_trend, fused_probs, fusion_weights, fusion_notes,
                    model_scores=model_scores,
                    handicap_expected=handicap_expected,
                    handicap_market_initial=handicap_market_initial,
                    handicap_market_final=handicap_market_final,
                    handicap_anomaly_label=handicap_anomaly_label,
                    handicap_anomaly_score=handicap_anomaly_score,
                    handicap_water_bias=handicap_water_bias,
                )
                
        except Exception as e:
            print(f"DeepSeek API调用失败: {str(e)}")
            if settings.require_ai_api:
                raise RuntimeError(f"AI API required but exception: {str(e)}")
            self.last_source = 'fallback'
            return self._generate_fallback_analysis(
                model_probs, market_probs, market_trend, fused_probs, fusion_weights, fusion_notes,
                model_scores=model_scores,
                handicap_expected=handicap_expected,
                handicap_market_initial=handicap_market_initial,
                handicap_market_final=handicap_market_final,
                handicap_anomaly_label=handicap_anomaly_label,
                handicap_anomaly_score=handicap_anomaly_score,
                handicap_water_bias=handicap_water_bias,
            )

    def _build_prompt(self,
                     home_team: TeamStats,
                     away_team: TeamStats,
                     model_probs: Dict[str, float],
                     model_scores: List[ScoreProbability],
                     market_probs: Dict[str, float],
                     market_trend: str,
                     history_summary: str = "",
                     market_history_summary: str = "",
                     market_history_stats: Dict = None,
                     fused_probs: Dict[str, float] = None,
                     fusion_weights: Dict = None,
                     fusion_notes: str = "",
                     handicap_expected: float = None,
                     handicap_market_initial: float = None,
                     handicap_market_final: float = None,
                     handicap_anomaly_label: str = "",
                     handicap_anomaly_score: float = None,
                     handicap_water_bias: str = "") -> str:
        """构建AI分析的prompt"""
        
        # —— 新增：高级指标（若提供）——
        gp_h = home_team.games_played if isinstance(home_team.games_played, int) and home_team.games_played > 0 else 1
        gp_a = away_team.games_played if isinstance(away_team.games_played, int) and away_team.games_played > 0 else 1

        def _fmt_total_avg(total: float, gp: int) -> str:
            try:
                return f"{total:.1f}（场均{(total/gp):.2f}）"
            except Exception:
                return "—"

        def _fmt_num(x: float) -> str:
            try:
                return f"{x:.2f}"
            except Exception:
                return "—"

        home_adv = []
        if isinstance(home_team.npxg, (int, float)):
            home_adv.append(f"npxG {_fmt_total_avg(home_team.npxg, gp_h)}")
        if isinstance(home_team.npxga, (int, float)):
            home_adv.append(f"npxGA {_fmt_total_avg(home_team.npxga, gp_h)}")
        if isinstance(home_team.ppda, (int, float)):
            home_adv.append(f"PPDA {_fmt_num(home_team.ppda)}")
        if isinstance(home_team.oppda, (int, float)):
            home_adv.append(f"OPPDA {_fmt_num(home_team.oppda)}")
        if isinstance(home_team.dc, (int, float)):
            home_adv.append(f"DC {_fmt_total_avg(home_team.dc, gp_h)}")
        if isinstance(home_team.odc, (int, float)):
            home_adv.append(f"ODC {_fmt_total_avg(home_team.odc, gp_h)}")

        away_adv = []
        if isinstance(away_team.npxg, (int, float)):
            away_adv.append(f"npxG {_fmt_total_avg(away_team.npxg, gp_a)}")
        if isinstance(away_team.npxga, (int, float)):
            away_adv.append(f"npxGA {_fmt_total_avg(away_team.npxga, gp_a)}")
        if isinstance(away_team.ppda, (int, float)):
            away_adv.append(f"PPDA {_fmt_num(away_team.ppda)}")
        if isinstance(away_team.oppda, (int, float)):
            away_adv.append(f"OPPDA {_fmt_num(away_team.oppda)}")
        if isinstance(away_team.dc, (int, float)):
            away_adv.append(f"DC {_fmt_total_avg(away_team.dc, gp_a)}")
        if isinstance(away_team.odc, (int, float)):
            away_adv.append(f"ODC {_fmt_total_avg(away_team.odc, gp_a)}")

        adv_block = ""
        if home_adv or away_adv:
            adv_block = (
                "\n高级指标（若提供）:\n"
                f"- 主队：{('，'.join(home_adv)) if home_adv else '—'}\n"
                f"- 客队：{('，'.join(away_adv)) if away_adv else '—'}\n"
            )

        # 历史段落（若未提供则省略）
        hist_block = ""
        if (history_summary or market_history_summary or (isinstance(market_history_stats, dict) and not market_history_stats.get('error'))):
            hist_block = (
                "\n历史维度:\n"
                f"- 摘要：{history_summary or '—'}\n"
                f"- 市场-历史对照：{market_history_summary or '—'}\n"
                f"- 分位统计：{self._format_stats_section(market_history_stats)}\n"
            )

        return f"""比赛：{home_team.name} vs {away_team.name}

数据总览:
- 主队（{home_team.games_played}场）：xG {home_team.xg:.1f}（场均{home_team.xg/home_team.games_played:.2f}），xGA {home_team.xga:.1f}（场均{home_team.xga/home_team.games_played:.2f}），xPTS {home_team.xpts:.1f}，赛季 进{home_team.goals_for} 失{home_team.goals_against}
- 客队（{away_team.games_played}场）：xG {away_team.xg:.1f}（场均{away_team.xg/away_team.games_played:.2f}），xGA {away_team.xga:.1f}（场均{away_team.xga/away_team.games_played:.2f}），xPTS {away_team.xpts:.1f}，赛季 进{away_team.goals_for} 失{away_team.goals_against}
{adv_block}模型输出:
- 概率：主胜{model_probs['home_win']:.1%}，平局{model_probs['draw']:.1%}，客胜{model_probs['away_win']:.1%}
- 可能比分：{', '.join(f"{s.score}({s.probability:.1%})" for s in model_scores)}

市场共识:
- 隐含概率：主胜{market_probs['home_win']:.1%}，平局{market_probs['draw']:.1%}，客胜{market_probs['away_win']:.1%}
- 趋势：{market_trend}

{hist_block}
模型-市场融合概率:
{self._format_fusion_section(fused_probs, fusion_weights, fusion_notes)}

让盘与水位:
- 模型期望让球：{f"{handicap_expected:+.2f}".replace("+0.00","0").replace("-0.00","0") if isinstance(handicap_expected,(int,float)) else "—"}
- 市场初盘→终盘均值：{f"{handicap_market_initial:+.2f}".replace("+0.00","0").replace("-0.00","0") if isinstance(handicap_market_initial,(int,float)) else "—"} → {f"{handicap_market_final:+.2f}".replace("+0.00","0").replace("-0.00","0") if isinstance(handicap_market_final,(int,float)) else "—"}
- 盘差(终盘-期望)：{f"{(handicap_market_final - handicap_expected):+.2f}".replace("+0.00","0").replace("-0.00","0") if isinstance(handicap_market_final,(int,float)) and isinstance(handicap_expected,(int,float)) else "—"}
- 水位偏向：{handicap_water_bias or '—'}
- 判定：{handicap_anomaly_label or "（数据不足）"}
- 异常分数：{f"{handicap_anomaly_score:.2f}" if isinstance(handicap_anomaly_score,(int,float)) else "—"}

风险与不确定性的解读:
- 数据缺失或偏差：如有则标注；无则“—”
- 潜在外生因素：不虚构，保守提示

结论:
- 基于以上要点给出专业结论。避免下注建议；若模型与市场共振，指出共振与稳定性；若相悖，解释偏差来源与风险。
"""

    def _generate_fallback_analysis(self,
                                  model_probs: Dict[str, float],
                                  market_probs: Dict[str, float],
                                  market_trend: str,
                                  fused_probs: Dict[str, float] = None,
                                  fusion_weights: Dict = None,
                                  fusion_notes: str = "",
                                  model_scores: List[ScoreProbability] = None,
                                  handicap_expected: float = None,
                                  handicap_market_initial: float = None,
                                  handicap_market_final: float = None,
                                  handicap_anomaly_label: str = "",
                                  handicap_anomaly_score: float = None,
                                  handicap_water_bias: str = "") -> str:
        """当API调用失败时生成基础分析报告"""
        
        # 计算模型和市场的观点差异
        prob_diff = {
            "home_win": model_probs["home_win"] - market_probs["home_win"],
            "draw": model_probs["draw"] - market_probs["draw"],
            "away_win": model_probs["away_win"] - market_probs["away_win"]
        }
        
        # 判断是否存在显著差异（差异超过5%视为显著）
        significant_diff = any(abs(diff) > 0.05 for diff in prob_diff.values())
        
        # 摘要：胜平负分布（优先融合）与比分预测
        dist = fused_probs if fused_probs else model_probs
        summary_line = (
            f"主胜{dist.get('home_win', 0):.1%} | 平局{dist.get('draw', 0):.1%} | 客胜{dist.get('away_win', 0):.1%}"
        )
        top_scores = "—"
        try:
            if model_scores:
                top3 = sorted(model_scores, key=lambda s: getattr(s, 'probability', 0), reverse=True)[:3]
                top_scores = ", ".join(f"{s.score}({s.probability:.1%})" for s in top3)
        except Exception:
            top_scores = "—"
        summary_block = f"摘要：\n- 胜平负分布：{summary_line}\n- 比分预测：{top_scores}\n"
        
        fused_text = ""
        if fused_probs and fusion_weights:
            fused_text = (
                f"\n融合后综合概率：主胜{fused_probs.get('home_win',0):.1%}、平局{fused_probs.get('draw',0):.1%}、客胜{fused_probs.get('away_win',0):.1%}。\n"
                f"权重：模型{fusion_weights.get('model',0):.2f}、市场{fusion_weights.get('market',0):.2f}。"
                + (f"\n说明：{fusion_notes}" if fusion_notes else "")
            )

        def _fmt_hcap(val):
            try:
                return f"{val:+.2f}".replace("+0.00","0").replace("-0.00","0")
            except Exception:
                return "—"
        delta_val = None
        try:
            if isinstance(handicap_market_final, (int,float)) and isinstance(handicap_expected, (int,float)):
                delta_val = handicap_market_final - handicap_expected
        except Exception:
            delta_val = None
        hcap_text = (
            "\n盘口判定：" +
            f"模型期望{_fmt_hcap(handicap_expected)}；初盘→终盘 {_fmt_hcap(handicap_market_initial)} → {_fmt_hcap(handicap_market_final)}；" +
            f"盘差(终盘-期望) {_fmt_hcap(delta_val)}；水位偏向：{handicap_water_bias or '—'}；" +
            f"判定：{handicap_anomaly_label or '（数据不足）'}；异常分数：{(f'{handicap_anomaly_score:.2f}' if isinstance(handicap_anomaly_score,(int,float)) else '—')}"
        )

        if significant_diff:
            return f"""分析报告：

{summary_block}
根据数据分析，我们的定量模型与市场共识存在一定差异。模型给出主胜{model_probs['home_win']:.1%}、平局{model_probs['draw']:.1%}、客胜{model_probs['away_win']:.1%}的概率预测，而市场隐含概率为主胜{market_probs['home_win']:.1%}、平局{market_probs['draw']:.1%}、客胜{market_probs['away_win']:.1%}。

市场趋势显示：{market_trend}
{fused_text}
{hcap_text}

这种差异提示这场比赛可能存在一些模型无法捕捉的特殊因素，建议在做出决策时持谨慎态度，综合考虑其他因素。"""
        else:
            return f"""分析报告：

{summary_block}
这场比赛的定量模型预测与市场共识高度一致，这是一个强烈的信号。模型预测主胜{model_probs['home_win']:.1%}、平局{model_probs['draw']:.1%}、客胜{model_probs['away_win']:.1%}，市场隐含概率也很接近。

市场趋势显示：{market_trend}
{fused_text}
{hcap_text}

基本面与市场面的共振表明，这个预测具有较高的可信度。"""

    def _format_fusion_section(self, fused_probs: Dict[str, float], fusion_weights: Dict, fusion_notes: str) -> str:
        try:
            if not fused_probs or not fusion_weights:
                return "（融合层未启用或数据不足。）"
            return (
                f"综合概率：主胜{fused_probs.get('home_win',0):.1%}、平局{fused_probs.get('draw',0):.1%}、客胜{fused_probs.get('away_win',0):.1%}\n"
                f"权重：模型{fusion_weights.get('model',0):.2f}、市场{fusion_weights.get('market',0):.2f}\n"
                + (f"说明：{fusion_notes}" if fusion_notes else "")
            )
        except Exception:
            return "（融合层未启用或数据不足。）"

    def _format_stats_section(self, stats: Dict) -> str:
        try:
            if not stats or stats.get("error"):
                return "（历史分位统计不足或未启用。）"
            hf = stats.get("hist_final_avg", {})
            cf = stats.get("cur_final_avg", {})
            hp = stats.get("hist_percentiles", {})
            lines = []
            lines.append(
                f"历史终盘均值：主胜{hf.get('home',0):.1%}、平局{hf.get('draw',0):.1%}、客胜{hf.get('away',0):.1%}"
            )
            lines.append(
                f"当前终盘均值：主胜{cf.get('home',0):.1%}、平局{cf.get('draw',0):.1%}、客胜{cf.get('away',0):.1%}"
            )
            def fmt_pct(x: float) -> str:
                return f"{x:.1%}"
            home_p = hp.get("home", {})
            lines.append(
                f"主胜分位：P25={fmt_pct(home_p.get('p25',0))}, P50={fmt_pct(home_p.get('p50',0))}, P75={fmt_pct(home_p.get('p75',0))}"
            )
            return "\n".join(lines)
        except Exception:
            return "（历史分位统计不足或未启用。）"

    def _ensure_tail(self, analysis: str,
                      fused_probs: Dict[str, float],
                      fusion_weights: Dict,
                      fusion_notes: str,
                      handicap_expected: float,
                      handicap_market_initial: float,
                      handicap_market_final: float,
                      handicap_anomaly_label: str,
                      handicap_anomaly_score: float,
                      handicap_water_bias: str) -> str:
        try:
            txt = analysis or ""
            has_conclusion = ("结论" in txt) or ("建议" in txt)
            ok_end = txt.strip().endswith(("。", "！", "?", "？"))
            if has_conclusion and ok_end:
                return txt

            fused_text = ""
            try:
                if fused_probs and fusion_weights:
                    fused_text = (
                        f"融合后综合概率：主胜{fused_probs.get('home_win',0):.1%}、平局{fused_probs.get('draw',0):.1%}、客胜{fused_probs.get('away_win',0):.1%}。\n"
                        f"权重：模型{fusion_weights.get('model',0):.2f}、市场{fusion_weights.get('market',0):.2f}。"
                        + (f"\n说明：{fusion_notes}" if fusion_notes else "")
                    )
            except Exception:
                fused_text = "（融合层未启用或数据不足。）"

            def _fmt_hcap(val):
                try:
                    return f"{val:+.2f}".replace("+0.00","0").replace("-0.00","0")
                except Exception:
                    return "—"

            delta_val = None
            try:
                if isinstance(handicap_market_final,(int,float)) and isinstance(handicap_expected,(int,float)):
                    delta_val = handicap_market_final - handicap_expected
            except Exception:
                delta_val = None
            hcap_text = (
                f"盘口判定：模型期望{_fmt_hcap(handicap_expected)}；初盘→终盘 {_fmt_hcap(handicap_market_initial)} → {_fmt_hcap(handicap_market_final)}；"
                f"盘差(终盘-期望) {_fmt_hcap(delta_val)}；水位偏向：{handicap_water_bias or '—'}；判定：{handicap_anomaly_label or '（数据不足）'}；异常分数："
                f"{(f'{handicap_anomaly_score:.2f}' if isinstance(handicap_anomaly_score,(int,float)) else '—')}"
            )
            tail = (
                "\n补充结论（为保证报告完整性）:\n"
                + (fused_text + "\n" if fused_text else "")
                + hcap_text + "\n"
                "风险提示：若历史统计不足或市场漂移较大，应保守解读。本段为本地补全，不含下注建议。"
            )
            return (txt + "\n" + tail).strip()
        except Exception:
            return analysis

    def generate_structured_output(self,
                                   home_team: TeamStats,
                                   away_team: TeamStats,
                                   model_probs: Dict[str, float],
                                   model_scores: List[ScoreProbability],
                                   market_probs: Dict[str, float],
                                   market_trend: str,
                                   history_summary: str = "",
                                   market_history_summary: str = "",
                                   market_history_stats: Dict = None,
                                   fused_probs: Dict[str, float] = None,
                                   fusion_weights: Dict = None,
                                   fusion_notes: str = "",
                                   handicap_expected: float = None,
                                   handicap_market_initial: float = None,
                                   handicap_market_final: float = None,
                                   handicap_anomaly_label: str = "",
                                   handicap_anomaly_score: float = None,
                                   handicap_water_bias: str = "",
                                   upset_team: str = "",
                                   upset_win_prob: float = None,
                                   upset_score: float = None,
                                   upset_label: str = "",
                                   upset_delta_vs_market: float = None) -> Dict:
        """生成结构化分析JSON（不依赖外部AI）。"""
        dist = fused_probs or model_probs or {}
        def _pct(x):
            try:
                return round(float(x) * 100.0, 1)
            except Exception:
                return None
        summary = {
            "match": f"{home_team.name} vs {away_team.name}",
            "distribution": {
                "home": _pct(dist.get("home_win")),
                "draw": _pct(dist.get("draw")),
                "away": _pct(dist.get("away_win")),
                "source": "fused" if fused_probs else "model"
            },
            "top_scores": [
                {"score": s.score, "prob": _pct(s.probability)} for s in (model_scores or [])[:3]
            ]
        }

        def _fmt_prob_section(title: str, probs: Dict[str, float]) -> Dict:
            return {
                "title": title,
                "items": [
                    {"type": "bullet", "text": f"主胜 { _pct(probs.get('home_win')) }%"},
                    {"type": "bullet", "text": f"平局 { _pct(probs.get('draw')) }%"},
                    {"type": "bullet", "text": f"客胜 { _pct(probs.get('away_win')) }%"},
                ]
            }

        sections = []
        sections.append({
            "title": "摘要",
            "items": [
                {"type": "paragraph", "text": f"胜/平/负分布（{summary['distribution']['source']}）：主胜{summary['distribution']['home']}%，平局{summary['distribution']['draw']}%，客胜{summary['distribution']['away']}%。"},
                {"type": "bullet", "text": ", ".join([f"{x['score']}({x['prob']}%)" for x in summary["top_scores"]]) or "—", "meta": {"label": "前3个可能比分"}},
            ]
        })

        sections.append(_fmt_prob_section("模型概率", model_probs or {}))
        sections.append({
            "title": "市场共识",
            "items": _fmt_prob_section("市场概率", market_probs or {})["items"] + [
                {"type": "paragraph", "text": f"趋势：{market_trend or '—'}"}
            ]
        })

        if fused_probs:
            sections.append({
                "title": "融合概率与权重",
                "items": _fmt_prob_section("融合概率", fused_probs)["items"] + [
                    {"type": "bullet", "text": f"权重：模型{round((fusion_weights or {}).get('model', 0.0)*100)}%，市场{round((fusion_weights or {}).get('market', 0.0)*100)}%"},
                    {"type": "paragraph", "text": f"说明：{fusion_notes or '—'}"}
                ]
            })

        hcap_items = []
        if handicap_expected is not None:
            hcap_items.append({"type": "bullet", "text": f"模型预期：让球 {handicap_expected}"})
        if handicap_market_initial is not None:
            hcap_items.append({"type": "bullet", "text": f"初盘：让球 {handicap_market_initial}"})
        if handicap_market_final is not None:
            hcap_items.append({"type": "bullet", "text": f"终盘：让球 {handicap_market_final}"})
        if handicap_anomaly_label:
            hcap_items.append({"type": "bullet", "text": f"异象：{handicap_anomaly_label}（score={handicap_anomaly_score}）"})
        if handicap_water_bias:
            hcap_items.append({"type": "bullet", "text": f"水位偏向：{handicap_water_bias}"})
        if hcap_items:
            sections.append({"title": "让盘与水位", "items": hcap_items})

        # 冷门与爆冷风险
        upset_items = []
        if upset_team:
            upset_items.append({"type": "bullet", "text": f"下狗：{upset_team}"})
        if isinstance(upset_win_prob, (int, float)):
            upset_items.append({"type": "bullet", "text": f"融合胜率：{_pct(upset_win_prob)}%"})
        if isinstance(upset_delta_vs_market, (int, float)):
            sign = "+" if upset_delta_vs_market >= 0 else "-"
            upset_items.append({"type": "bullet", "text": f"模型-市场差：{sign}{_pct(abs(upset_delta_vs_market))}%"})
        if isinstance(upset_score, (int, float)):
            try:
                upset_items.append({"type": "bullet", "text": f"风险评分：{round(upset_score*100)}分（{upset_label or '—'}）"})
            except Exception:
                pass
        if upset_items:
            sections.append({"title": "冷门与爆冷风险", "items": upset_items})

        hist_items = []
        if history_summary:
            hist_items.append({"type": "paragraph", "text": f"历史摘要：{history_summary}"})
        if market_history_summary:
            hist_items.append({"type": "paragraph", "text": f"市场-历史对照：{market_history_summary}"})
        if isinstance(market_history_stats, dict) and not market_history_stats.get("error"):
            hist_items.append({"type": "metric", "text": "分位统计", "meta": market_history_stats})
        if hist_items:
            sections.append({"title": "历史与阵容", "items": hist_items})

        sections.append({
            "title": "风险与结论",
            "items": [
                {"type": "paragraph", "text": "分析为信息参考，不构成下注建议。模型可能存在样本不足与市场异动导致的不确定性。"},
                {"type": "paragraph", "text": "建议关注临场阵容与伤病、盘口临盘变化，以动态更新判断。"},
            ]
        })

        return {
            "format_version": "v1",
            "summary": summary,
            "sections": sections
        }