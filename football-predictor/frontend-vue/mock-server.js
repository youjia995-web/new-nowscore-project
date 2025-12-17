import express from 'express';
import cors from 'cors';

const app = express();
const PORT = 8000;

app.use(cors());
app.use(express.json());

// Mock team metrics data
const teamMetrics = {
  'Napoli': {
    team: 'Napoli',
    season: '2025/2026',
    games_played: 10,
    points: 22,
    wins: 7,
    draws: 1,
    losses: 2,
    goals_for: 16,
    goals_against: 10,
    xg: 16.93,
    xga: 14.32,
    xpts: 16.68,
    ppda: 10.03,
    dc: 73.0
  }
};

// API endpoint for team metrics
app.get('/api/teams/metrics', (req, res) => {
  const { team } = req.query;
  console.log(`Received request for team: ${team}`);
  
  if (team && teamMetrics[team]) {
    console.log(`Returning data for ${team}:`, teamMetrics[team]);
    res.json(teamMetrics[team]);
  } else {
    console.log(`Team not found: ${team}`);
    res.status(404).json({ error: 'Team not found' });
  }
});

// API endpoint for prediction
app.post('/api/predict', (req, res) => {
  console.log('Received prediction request:', req.body);
  
  // Mock prediction response
  const mockResponse = {
    model_home_win: 0.45,
    model_draw: 0.25,
    model_away_win: 0.30,
    model_likely_scores: ['2-1', '1-1', '1-0'],
    market_home_win: 0.42,
    market_draw: 0.28,
    market_away_win: 0.30,
    fused_home_win: 0.44,
    fused_draw: 0.26,
    fused_away_win: 0.30,
    ai_analysis: '基于数据分析，主队稍占优势。'
  };
  
  res.json(mockResponse);
});

// API endpoint for runtime config
app.get('/api/admin/config', (req, res) => {
  console.log('Received config request');
  
  const mockConfig = {
    require_ai_api: false,
    gru_enabled_leagues: 'Premier League,La Liga,Serie A,Bundesliga,Ligue 1',
    gru_model_weight: 0.3,
    ai_key_present: false,
    ai_api_base: ''
  };
  
  res.json(mockConfig);
});

app.listen(PORT, () => {
  console.log(`Mock server running on http://localhost:${PORT}`);
  console.log('Available endpoints:');
  console.log('  GET  /api/teams/metrics?team=Napoli');
  console.log('  POST /api/predict');
  console.log('  GET  /api/admin/config');
});