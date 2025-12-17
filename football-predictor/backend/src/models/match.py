from pydantic import BaseModel
from typing import List, Optional, Dict

class TeamStats(BaseModel):
    name: str
    ranking: int # 球队在联赛中的当前排名
    games_played: int  # 本赛季已进行的比赛场次
    xg: float  # 本赛季至今的累积预期进球总和
    xga: float  # 本赛季至今的累积预期失球总和
    xpts: float  # 本赛季至今的累积预期积分总和
    # 当前比赛采用的阵型（前端手动填入，如"4-2-3-1"）
    formation: Optional[str] = None
    # 赛季累计进球/失球（优先使用）
    goals_for: Optional[int] = None
    goals_against: Optional[int] = None
    # 近6场总进球/失球（可选，作为备用短期参考）
    recent_goals_scored: Optional[int] = None
    recent_goals_conceded: Optional[int] = None
    npxg: Optional[float] = None
    npxga: Optional[float] = None
    ppda: Optional[float] = None
    oppda: Optional[float] = None
    dc: Optional[float] = None
    odc: Optional[float] = None

class OddsData(BaseModel):
    company: str
    initial_home_win: float
    initial_draw: float
    initial_away_win: float
    final_home_win: float
    final_draw: float
    final_away_win: float
    initial_handicap: str
    final_handicap: str
    initial_handicap_home_odds: float
    initial_handicap_away_odds: float
    final_handicap_home_odds: float
    final_handicap_away_odds: float

class RosterItem(BaseModel):
    player_name: str
    appearances: Optional[int] = None
    starts: Optional[int] = None
    minutes: Optional[int] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    rating: Optional[float] = None
    market_value: Optional[float] = None
    injured: Optional[bool] = None
    suspended: Optional[bool] = None

class MatchInput(BaseModel):
    home_team: TeamStats
    away_team: TeamStats
    odds_data: List[OddsData]
    # 临时阵容（仅用于当前比赛的可用性调整）
    home_roster: Optional[List[RosterItem]] = None
    away_roster: Optional[List[RosterItem]] = None
    # 前端联赛参数建议选择（可选）："英超/德甲"、"意甲/法甲"、"西甲"
    league_preset: Optional[str] = None

class ScoreProbability(BaseModel):
    score: str
    probability: float

class PredictionResult(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    likely_scores: List[ScoreProbability]
    market_home_win_prob: float
    market_draw_prob: float
    market_away_win_prob: float
    market_trend: str
    analysis: str
    history_summary: Optional[str] = None
    market_history_summary: Optional[str] = None
    # 结构化历史统计（用于前端图表），字段含均值与分位点
    market_history_stats: Optional[Dict] = None
    # 模型-市场融合后的综合概率
    fused_home_win_prob: Optional[float] = None
    fused_draw_prob: Optional[float] = None
    fused_away_win_prob: Optional[float] = None
    # 融合权重与说明（便于前端展示透明度）
    fusion_weights: Optional[Dict] = None
    fusion_notes: Optional[str] = None
    # 盘口合理性与异象识别
    expected_handicap: Optional[float] = None
    market_final_handicap: Optional[float] = None
    handicap_anomaly_score: Optional[float] = None
    handicap_anomaly_label: Optional[str] = None
    handicap_water_bias: Optional[str] = None
    # 比分概率矩阵（用于前端热力图），以及矩阵轴最大进球数
    score_matrix: Optional[List[List[float]]] = None
    matrix_goals_max: Optional[int] = None
    # 校准与一致性指标（用于评估模型与市场的距离/锐度等）
    calibration_metrics: Optional[Dict[str, float]] = None
    calibration_notes: Optional[str] = None
    # 标记 AI 分析来源：'deepseek' 或 'fallback'
    analysis_source: Optional[str] = None
    # 结构化分析输出（供前端渲染器）：分区化要点与摘要
    analysis_struct: Optional[Dict] = None

    # —— 冷门因子（爆冷风险） ——
    upset_team: Optional[str] = None
    upset_win_prob: Optional[float] = None  # 使用融合或模型的该队胜率
    upset_score: Optional[float] = None     # 归一化[0,1]
    upset_label: Optional[str] = None       # 文本标签：高/中/低
    upset_notes: Optional[str] = None       # 依据与理由摘要

class AnalysisItem(BaseModel):
    # item 类型：bullet/paragraph/metric
    type: str
    text: str
    meta: Optional[Dict] = None

class AnalysisSection(BaseModel):
    title: str
    items: List[AnalysisItem]

class AnalysisOutput(BaseModel):
    format_version: str = "v1"
    summary: Optional[Dict] = None
    sections: List[AnalysisSection]

class TeamSeasonStats(BaseModel):
    team: str
    season: str
    xg: float
    xga: float
    xpts: float
    notes: Optional[str] = None

class AdminMatch(BaseModel):
    date: Optional[str] = None
    league: Optional[str] = None
    season: Optional[str] = None
    home_team: str
    away_team: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    # —— 扩展：原库字段与盘口/赔率 ——
    round: Optional[str] = None
    home_ranking: Optional[int] = None
    away_ranking: Optional[int] = None
    half_home_goals: Optional[int] = None
    half_away_goals: Optional[int] = None
    odds_data: Optional[List[OddsData]] = None
    notes: Optional[str] = None

class TeamDailyMetrics(BaseModel):
    team: str
    season: str
    date: str
    league: Optional[str] = None
    points: Optional[float] = None
    wins: Optional[int] = None
    draws: Optional[int] = None
    losses: Optional[int] = None
    goals_for: Optional[int] = None
    goals_against: Optional[int] = None
    xg: Optional[float] = None
    npxg: Optional[float] = None
    xga: Optional[float] = None
    npxga: Optional[float] = None
    xpxgd: Optional[float] = None
    ppda: Optional[float] = None
    oppda: Optional[float] = None
    dc: Optional[float] = None
    odc: Optional[float] = None
    xpts: Optional[float] = None