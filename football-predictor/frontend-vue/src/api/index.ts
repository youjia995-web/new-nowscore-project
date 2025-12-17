import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.DEV ? '/api' : '/api',
  timeout: 120000,
})

export interface PredictReq {
  home_team: {
    name:string
    formation?:string
    ranking:number
    games_played:number
    xg:number
    xga:number
    xpts:number
    season_goals_for:number
    season_goals_against:number
    npxg?:number
    npxga?:number
    ppda?:number
    oppda?:number
    dc?:number
    odc?:number
    wins?:number
    draws?:number
    losses?:number
    points?:number
    xpxgd?:number
    roster?:{name:string,position:string,rating:number}[]
  }
  away_team: {
    name:string
    formation?:string
    ranking:number
    games_played:number
    xg:number
    xga:number
    xpts:number
    season_goals_for:number
    season_goals_against:number
    npxg?:number
    npxga?:number
    ppda?:number
    oppda?:number
    dc?:number
    odc?:number
    wins?:number
    draws?:number
    losses?:number
    points?:number
    xpxgd?:number
    roster?:{name:string,position:string,rating:number}[]
  }
  league_preset?: '英超/德甲'|'意甲/法甲'|'西甲'
  odds_data:{
    company:string
    initial_home_win?:number
    initial_draw?:number
    initial_away_win?:number
    initial_handicap:string
    initial_handicap_home_odds:number
    initial_handicap_away_odds:number
    final_home_win?:number
    final_draw?:number
    final_away_win?:number
    final_handicap:string
    final_handicap_home_odds:number
    final_handicap_away_odds:number
  }[]
}

export interface PredictResp {
  home_win_prob?: number
  draw_prob?: number
  away_win_prob?: number
  likely_scores?: { score: string }[]
  market_home_win_prob?: number
  market_draw_prob?: number
  market_away_win_prob?: number
  market_trend?: string
  fused_home_win_prob?: number
  fused_draw_prob?: number
  fused_away_win_prob?: number
  fusion_weights?: { model?: number, market?: number }
  fusion_notes?: string
  expected_handicap?: number
  market_final_handicap?: number
  handicap_anomaly_label?: string
  handicap_anomaly_score?: number
  handicap_water_bias?: string
  upset_team?: string
  upset_win_prob?: number
  upset_score?: number
  upset_label?: string
  upset_notes?: string
  score_matrix?: number[][]
  calibration_curve?: { pred: number, actual: number }[]
  backtest_summary?: string
  analysis?: string
  analysis_struct?: { summary?: unknown }
}

export interface ConfigItem {
  key: string
  value: string | number | boolean
}

export interface AIStatus {
  status: 'ok' | 'error'
  message?: string
}

export const predict = (data: PredictReq) =>
  api.post<PredictResp>('/predict', data).then(r => r.data)

export interface RuntimeConfig {
  require_ai_api: boolean
  gru_enabled_leagues: string
  gru_model_weight: number
  ai_key_present: boolean
  ai_api_base: string
}

export const getConfig = () =>
  api.get<RuntimeConfig>('/admin/config').then(r => r.data)

export const setConfig = (data: ConfigItem[] | Record<string, unknown>) =>
  api.post('/admin/config', data).then(r => r.data)

export interface TeamMetrics {
  team: string
  season?: string
  games_played: number
  points: number
  wins: number
  draws: number
  losses: number
  goals_for: number
  goals_against: number
  xg: number
  npxg?: number
  xga: number
  npxga?: number
  xpxgd?: number
  ppda?: number
  oppda?: number
  dc?: number
  odc?: number
  xpts: number
}

export const getTeamMetrics = (team: string, season?: string) =>
  api.get<TeamMetrics>('/teams/metrics', { params: { team, season } }).then(r => r.data)

export const getAIStatus = () =>
  api.get<AIStatus>('/admin/ai-status').then(r => r.data)

export const listCompanies = () =>
  api.get<{ companies: string[] }>('/companies').then(r => r.data)

export interface BacktestBin {
  bin: string
  mean_prob: number | null
  accuracy: number | null
  count: number
}

export interface BacktestSummary {
  samples: number
  accuracy: number | null
  avg_brier: number | null
  avg_logloss: number | null
  reliability: BacktestBin[]
  ece: number | null
}

export interface BacktestCalibration {
  samples: number
  per_class_accuracy: { home: number | null, draw: number | null, away: number | null }
  reliability_by_class: { home: BacktestBin[], draw: BacktestBin[], away: BacktestBin[] }
  ece_by_class: { home: number | null, draw: number | null, away: number | null }
  ece_macro: number | null
}

export const getBacktestSummary = () =>
  api.get<BacktestSummary>('/backtest/summary').then(r => r.data)

export const getBacktestCalibration = () =>
  api.get<BacktestCalibration>('/backtest/calibration').then(r => r.data)

export interface PredictionLog {
  id: number
  created_at: string
  home_team: string
  away_team: string
  pred_outcome: string
  pred_confidence: string
  pred_probs: { home: number, draw: number, away: number }
  market_probs: { home: number, draw: number, away: number }
  likely_score?: string | null
  calibration?: { brier?: number | null, kl?: number | null }
  analysis_source?: string | null
  actual?: { home_goals: number, away_goals: number, outcome: string } | null
}

export const listPredictionsByDate = (date: string, opts?: { team?: string, has_result?: boolean }) =>
  api.get<PredictionLog[]>('/predictions/daily', { params: { date, team: opts?.team, has_result: opts?.has_result } }).then(r => r.data)

export const getCalendarCounts = (month: string) =>
  api.get<{ date: string, count: number }[]>('/predictions/calendar', { params: { month } }).then(r => r.data)

export const getPrediction = (id: number) =>
  api.get('/predictions/' + id).then(r => r.data)

export const updateActualResult = (id: number, home_goals: number, away_goals: number) =>
  api.patch<{ id: number, actual_outcome: string }>(`/predictions/${id}/result`, { home_goals, away_goals }).then(r => r.data)

export const autoFillActualResult = (id: number) =>
  api.post<{ id?: number, actual_outcome?: string, found: boolean }>(`/predictions/${id}/auto-result`).then(r => r.data)

export default api
