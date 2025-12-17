<template>
  <div class="page">
    <h1 class="text-2xl font-bold mb-4">比赛预测</h1>

    <!-- 球队数据 -->
    <div class="kf-card mb-6">
      <div class="kf-card-header">
        <div class="kf-card-title">球队数据</div>
        <div class="kf-card-actions">
          <el-button size="small" type="warning" @click="testAutoFill">测试自动填充</el-button>
        </div>
      </div>
      <el-form :model="form" label-position="top" class="kf-form">
      <el-row :gutter="16">
        <!-- 主队 -->
        <el-col :span="12">
          <div class="kf-subtitle">主队</div>
          <el-form-item label="球队名称">
            <el-input v-model="form.home_team.name" placeholder="必填" style="max-width: 320px;" />
            <el-button
              size="small"
              type="primary"
              :disabled="!form.home_team.name || !form.home_team.name.trim()"
              @click="autoFillTeamData('home')"
              style="margin-left: 8px;"
            >填充</el-button>
          </el-form-item>
          <el-form-item label="阵型"><el-input v-model="form.home_team.formation" placeholder="如 4-2-3-1" style="max-width: 240px;" /></el-form-item>
          <el-row :gutter="12">
            <el-col :span="12"><el-form-item label="当前排名"><el-input-number v-model="form.home_team.ranking" :min="1" required /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="已赛场次"><el-input-number v-model="form.home_team.games_played" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="xG"><el-input-number v-model="form.home_team.xg" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="xGA"><el-input-number v-model="form.home_team.xga" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="xPTS"><el-input-number v-model="form.home_team.xpts" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="赛季进球"><el-input-number v-model="form.home_team.season_goals_for" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="赛季失球"><el-input-number v-model="form.home_team.season_goals_against" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="npxG"><el-input-number v-model="form.home_team.npxg" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="npxGA"><el-input-number v-model="form.home_team.npxga" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="PPDA"><el-input-number v-model="form.home_team.ppda" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="OPPDA"><el-input-number v-model="form.home_team.oppda" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="DC"><el-input-number v-model="form.home_team.dc" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="ODC"><el-input-number v-model="form.home_team.odc" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="胜"><el-input-number v-model="form.home_team.wins" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="平"><el-input-number v-model="form.home_team.draws" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="负"><el-input-number v-model="form.home_team.losses" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="积分"><el-input-number v-model="form.home_team.points" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="NPxGD"><el-input-number v-model="form.home_team.xpxgd" :step="0.01" :precision="2" /></el-form-item></el-col>
          </el-row>
          <el-form-item>
            <el-button size="small" @click="openRosterModal('home')">编辑阵容</el-button>
            <span class="ml-2 text-sm text-gray-500">{{ homeRosterSummary }}</span>
          </el-form-item>
        </el-col>
        <!-- 客队 -->
        <el-col :span="12">
          <div class="kf-subtitle">客队</div>
          <el-form-item label="球队名称">
            <el-input v-model="form.away_team.name" placeholder="必填" style="max-width: 320px;" />
            <el-button
              size="small"
              type="primary"
              :disabled="!form.away_team.name || !form.away_team.name.trim()"
              @click="autoFillTeamData('away')"
              style="margin-left: 8px;"
            >填充</el-button>
          </el-form-item>
          <el-form-item label="阵型"><el-input v-model="form.away_team.formation" placeholder="如 4-3-3" style="max-width: 240px;" /></el-form-item>
          <el-row :gutter="12">
            <el-col :span="12"><el-form-item label="当前排名"><el-input-number v-model="form.away_team.ranking" :min="1" required /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="已赛场次"><el-input-number v-model="form.away_team.games_played" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="xG"><el-input-number v-model="form.away_team.xg" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="xGA"><el-input-number v-model="form.away_team.xga" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="xPTS"><el-input-number v-model="form.away_team.xpts" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="赛季进球"><el-input-number v-model="form.away_team.season_goals_for" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="赛季失球"><el-input-number v-model="form.away_team.season_goals_against" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="npxG"><el-input-number v-model="form.away_team.npxg" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="npxGA"><el-input-number v-model="form.away_team.npxga" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="PPDA"><el-input-number v-model="form.away_team.ppda" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="OPPDA"><el-input-number v-model="form.away_team.oppda" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="DC"><el-input-number v-model="form.away_team.dc" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="ODC"><el-input-number v-model="form.away_team.odc" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="胜"><el-input-number v-model="form.away_team.wins" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="平"><el-input-number v-model="form.away_team.draws" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="负"><el-input-number v-model="form.away_team.losses" :min="0" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="积分"><el-input-number v-model="form.away_team.points" :step="0.01" :precision="2" /></el-form-item></el-col>
            <el-col :span="12"><el-form-item label="NPxGD"><el-input-number v-model="form.away_team.xpxgd" :step="0.01" :precision="2" /></el-form-item></el-col>
          </el-row>
          <el-form-item>
            <el-button size="small" @click="openRosterModal('away')">编辑阵容</el-button>
            <span class="ml-2 text-sm text-gray-500">{{ awayRosterSummary }}</span>
          </el-form-item>
        </el-col>
      </el-row>
      </el-form>
    </div>

    <div class="kf-card mb-6">
      <div class="kf-card-header">
        <div class="kf-card-title">联赛参数建议</div>
      </div>
      <el-form :model="form" label-position="top" class="kf-form">
        <el-form-item label="联赛选择">
          <el-select v-model="form.league_preset" placeholder="不使用联赛建议">
            <el-option label="英超/德甲" value="英超/德甲" />
            <el-option label="意甲/法甲" value="意甲/法甲" />
            <el-option label="西甲" value="西甲" />
          </el-select>
          <div class="kf-help">自动应用 BP_SHARED_STRENGTH 与 DRAW_BOOST_STRENGTH 建议值（仅当前请求）。</div>
        </el-form-item>
      </el-form>
    </div>

    <div class="kf-card mb-6">
      <div class="kf-card-header">
        <div class="kf-card-title">赔率数据</div>
        <div class="kf-card-actions"><el-button size="small" @click="addOddsRow">+ 添加公司</el-button></div>
      </div>
      <div class="kf-odds-list">
        <div v-for="(row, idx) in form.odds" :key="idx" class="kf-odds-row">
          <div class="flex items-center mb-2">
            <el-select v-model="row.company" placeholder="选择公司" style="max-width: 220px;">
              <el-option v-for="c in companyOptions" :key="c" :label="c" :value="c" />
            </el-select>
            <el-button size="small" type="danger" text style="margin-left: auto;" @click="removeOddsRow(idx)">删除</el-button>
          </div>
          <div class="text-sm text-gray-600 mb-1">胜平负</div>
          <div class="kf-odds-split mb-2">
            <div>
              <div class="text-xs text-gray-500 mb-1">初盘</div>
              <div class="kf-odds-grid">
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">主胜</div>
                  <el-input-number v-model="row.initial_home_win" :step="0.01" :precision="2" :min="1" />
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">平</div>
                  <el-input-number v-model="row.initial_draw" :step="0.01" :precision="2" :min="1" />
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">客胜</div>
                  <el-input-number v-model="row.initial_away_win" :step="0.01" :precision="2" :min="1" />
                </div>
              </div>
            </div>
            <div>
              <div class="text-xs text-gray-500 mb-1">终盘</div>
              <div class="kf-odds-grid">
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">主胜</div>
                  <el-input-number v-model="row.final_home_win" :step="0.01" :precision="2" :min="1" />
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">平</div>
                  <el-input-number v-model="row.final_draw" :step="0.01" :precision="2" :min="1" />
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">客胜</div>
                  <el-input-number v-model="row.final_away_win" :step="0.01" :precision="2" :min="1" />
                </div>
              </div>
            </div>
          </div>
          <div class="text-sm text-gray-600 mb-1">让球盘</div>
          <div class="kf-odds-split">
            <div>
              <div class="text-xs text-gray-500 mb-1">初盘</div>
              <div class="kf-odds-grid3">
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">盘口</div>
                  <el-select v-model="row.initial_handicap" placeholder="盘口" style="width: 120px;">
                    <el-option v-for="h in handicapOpts" :key="h" :label="h" :value="h" />
                  </el-select>
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">主盘</div>
                  <el-input-number v-model="row.initial_handicap_home_odds" :step="0.01" :precision="2" :min="0.01" />
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">客盘</div>
                  <el-input-number v-model="row.initial_handicap_away_odds" :step="0.01" :precision="2" :min="0.01" />
                </div>
              </div>
            </div>
            <div>
              <div class="text-xs text-gray-500 mb-1">终盘</div>
              <div class="kf-odds-grid3">
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">盘口</div>
                  <el-select v-model="row.final_handicap" placeholder="盘口" style="width: 120px;">
                    <el-option v-for="h in handicapOpts" :key="h" :label="h" :value="h" />
                  </el-select>
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">主盘</div>
                  <el-input-number v-model="row.final_handicap_home_odds" :step="0.01" :precision="2" :min="0.01" />
                </div>
                <div class="kf-odds-cell">
                  <div class="kf-odds-label">客盘</div>
                  <el-input-number v-model="row.final_handicap_away_odds" :step="0.01" :precision="2" :min="0.01" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="kf-card mb-6">
      <div class="kf-card-header">
        <div class="kf-card-title">操作</div>
      </div>
      <div class="kf-actions">
        <el-button type="primary" :loading="loading" :disabled="loading" @click="submitPredict">提交预测</el-button>
        <el-button @click="loadMockData">加载模拟数据</el-button>
      </div>
    </div>

    <el-card v-if="progressVisible" class="mb-4" shadow="never">
      <template #header><div class="font-semibold">预测进度</div></template>
      <div class="pixel-spinner">
        <div class="sq"></div>
        <div class="sq"></div>
        <div class="sq"></div>
      </div>
      <div class="mt-2 text-sm text-gray-600">{{ progressText }}</div>
      <div class="mt-3 text-right">
        <el-button size="small" @click="cancelProgress">取消/重置</el-button>
      </div>
    </el-card>

    <!-- 结果展示 -->
    <div v-if="result" class="space-y-4">
      <div class="kf-match">
        <div class="kf-teams">
          <div class="kf-team"><div class="kf-team-name">{{ form.home_team.name || '主队' }}</div></div>
          <div class="kf-vs">vs</div>
          <div class="kf-team"><div class="kf-team-name">{{ form.away_team.name || '客队' }}</div></div>
        </div>
        <div class="kf-meta">
          <span class="kf-meta-item">联赛：{{ form.league_preset || '—' }}</span>
          <span class="kf-meta-sep">•</span>
          <span class="kf-meta-item">公司数 {{ form.odds.length }}</span>
          <span v-if="result && result.market_trend" class="kf-meta-item">市场趋势：{{ result.market_trend }}</span>
        </div>
      </div>
      <div class="kf-section">
        <div class="kf-dist">
          <div class="kf-item home">
            <div class="kf-num">{{ pc(result.fused_home_win_prob ?? result.home_win_prob) }}</div>
            <div class="kf-label">主胜</div>
            <div class="kf-bar"><span class="home" :style="{width: pctVal(result.fused_home_win_prob ?? result.home_win_prob)}"></span></div>
          </div>
          <div class="kf-item draw">
            <div class="kf-num">{{ pc(result.fused_draw_prob ?? result.draw_prob) }}</div>
            <div class="kf-label">平局</div>
            <div class="kf-bar"><span class="draw" :style="{width: pctVal(result.fused_draw_prob ?? result.draw_prob)}"></span></div>
          </div>
          <div class="kf-item away">
            <div class="kf-num">{{ pc(result.fused_away_win_prob ?? result.away_win_prob) }}</div>
            <div class="kf-label">客胜</div>
            <div class="kf-bar"><span class="away" :style="{width: pctVal(result.fused_away_win_prob ?? result.away_win_prob)}"></span></div>
          </div>
        </div>
        <div class="kf-top">
          <div class="kf-top-title">最可能比分</div>
          <div class="kf-top-list">
            <span v-for="s in topScores" :key="s.score">{{ s.score }} {{ pc((s as any).prob) }}</span>
          </div>
        </div>
        <div class="kf-pairs">
          <div class="kf-pair">
            <div class="kf-pair-title">Over 1.5 / Under 1.5</div>
            <div class="kf-pair-bars">
              <span class="left" :style="{width: pctVal(totals?.over15)}"></span>
              <span class="right" :style="{width: pctVal(totals?.under15)}"></span>
            </div>
            <div class="kf-pair-values"><span>{{ pc(totals?.over15) }}</span><span>{{ pc(totals?.under15) }}</span></div>
          </div>
          <div class="kf-pair">
            <div class="kf-pair-title">Over 2.5 / Under 2.5</div>
            <div class="kf-pair-bars">
              <span class="left" :style="{width: pctVal(totals?.over25)}"></span>
              <span class="right" :style="{width: pctVal(totals?.under25)}"></span>
            </div>
            <div class="kf-pair-values"><span>{{ pc(totals?.over25) }}</span><span>{{ pc(totals?.under25) }}</span></div>
          </div>
          <div class="kf-pair">
            <div class="kf-pair-title">Over 3.5 / Under 3.5</div>
            <div class="kf-pair-bars">
              <span class="left" :style="{width: pctVal(totals?.over35)}"></span>
              <span class="right" :style="{width: pctVal(totals?.under35)}"></span>
            </div>
            <div class="kf-pair-values"><span>{{ pc(totals?.over35) }}</span><span>{{ pc(totals?.under35) }}</span></div>
          </div>
          <div class="kf-pair">
            <div class="kf-pair-title">双方进球 (BTTS)</div>
            <div class="kf-pair-bars">
              <span class="left" :style="{width: pctVal(totals?.btts_yes)}"></span>
              <span class="right" :style="{width: pctVal(totals?.btts_no)}"></span>
            </div>
            <div class="kf-pair-values"><span>Yes {{ pc(totals?.btts_yes) }}</span><span>No {{ pc(totals?.btts_no) }}</span></div>
          </div>
        </div>
      </div>

      <!-- 让盘与冷门 -->
      <el-row :gutter="12">
        <el-col :span="12">
          <el-card class="mb-2">
            <template #header><div class="font-semibold">让盘合理性</div></template>
            <div class="kf-hcap">
              <div class="kf-metric">
                <div class="kf-metric-title">模型期望</div>
                <div class="kf-metric-value">{{ fmtHcap(result.expected_handicap) }}</div>
              </div>
              <div class="kf-metric">
                <div class="kf-metric-title">市场终盘</div>
                <div class="kf-metric-value">{{ fmtHcap(result.market_final_handicap) }}</div>
              </div>
              <div class="kf-metric">
                <div class="kf-metric-title">水位偏向</div>
                <div class="kf-chip">{{ result.handicap_water_bias || '—' }}</div>
              </div>
            </div>
            <div class="kf-bar-wrap">
              <div class="kf-bar"><span :class="['sev', hcapSev]" :style="{width: pctVal(result.handicap_anomaly_score)}"></span></div>
              <div class="kf-bar-text">
                <span>判定 {{ result.handicap_anomaly_label || '—' }}</span>
                <span>异常分数 {{ fmtScore(result.handicap_anomaly_score) }}</span>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card class="mb-2">
            <template #header><div class="font-semibold">冷门预警</div></template>
            <div class="kf-upset">
              <div class="kf-metric">
                <div class="kf-metric-title">下狗</div>
                <div class="kf-metric-value">{{ result.upset_team || '—' }}</div>
              </div>
              <div class="kf-metric">
                <div class="kf-metric-title">融合胜率</div>
                <div class="kf-metric-value">{{ pc(result.upset_win_prob) }}</div>
              </div>
              <div class="kf-chip">{{ result.upset_label || '—' }}</div>
              <div class="kf-bar"><span :class="['sev', upsetTag]" :style="{width: pctVal(result.upset_score)}"></span></div>
              <div class="kf-note">{{ result.upset_notes || '—' }}</div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 各公司隐含概率（必发交易快照） -->
      <el-card class="mt-4">
        <template #header><div class="font-semibold">各公司隐含概率（终盘）</div></template>
        <div v-if="impliedBarData.length > 0" class="kf-comp-grid">
          <div v-for="item in impliedBarData" :key="item.company" class="kf-company">
            <div class="kf-company-name">{{ item.company }}</div>
            <div class="kf-3bars">
              <div class="bar home" :style="{width: item.home + '%'}"><span>{{ item.home }}%</span></div>
              <div class="bar draw" :style="{width: item.draw + '%'}"><span>{{ item.draw }}%</span></div>
              <div class="bar away" :style="{width: item.away + '%'}"><span>{{ item.away }}%</span></div>
            </div>
          </div>
        </div>
        <div v-else class="text-sm text-gray-500">未提供公司赔率数据</div>
      </el-card>

      <!-- 比分热力图 -->
      <el-card class="mb-2">
        <template #header><div class="font-semibold">比分概率热力图</div></template>
        <div class="heatmap-wrap"><div ref="heatmapDom" class="heatmap" /></div>
        <div class="text-xs text-gray-500 mt-2">深色表示概率更高；坐标轴为主队/客队进球数。</div>
      </el-card>

      <!-- 校准曲线 -->
      <el-card class="mb-2">
        <template #header><div class="font-semibold">校准曲线与回测</div></template>
        <div ref="calibDom" style="height: 240px" />
        <div class="text-xs text-gray-500 mt-2">{{ result.backtest_summary || '—' }}</div>
      </el-card>

      <!-- AI 综合分析 -->
      <el-card class="mb-2">
        <template #header>
          <div class="flex justify-between items-center">
            <div class="font-semibold">AI 综合分析与建议</div>
            <el-radio-group v-model="aiShow" size="small">
              <el-radio-button label="text">文本</el-radio-button>
              <el-radio-button label="structured">结构化</el-radio-button>
              <el-radio-button label="json">JSON</el-radio-button>
            </el-radio-group>
          </div>
        </template>
        <div class="kf-ana">
          <pre v-if="aiShow==='text'" class="kf-pre">{{ result.analysis }}</pre>
          <pre v-else-if="aiShow==='structured'" class="kf-pre">{{ JSON.stringify(result.analysis_struct?.summary,null,2) }}</pre>
          <pre v-else class="kf-pre small">{{ JSON.stringify(result.analysis_struct,null,2) }}</pre>
        </div>
      </el-card>

      <!-- 图表：三路概率 -->
      <el-card>
        <template #header><div class="font-semibold">三路融合概率图表</div></template>
        <div ref="chartFusedDom" style="height: 380px" />
      </el-card>
    </div>

    <!-- 结果弹窗 -->
    <el-dialog v-model="resultDlg" title="预测结果" width="520px">
      <div class="space-y-2 text-sm">
        <div>综合概率：主胜 {{ pc(result?.fused_home_win_prob) }} | 平局 {{ pc(result?.fused_draw_prob) }} | 客胜 {{ pc(result?.fused_away_win_prob) }}</div>
        <div>模型概率：主胜 {{ pc(result?.home_win_prob) }} | 平局 {{ pc(result?.draw_prob) }} | 客胜 {{ pc(result?.away_win_prob) }}</div>
        <div>市场概率：主胜 {{ pc(result?.market_home_win_prob) }} | 平局 {{ pc(result?.market_draw_prob) }} | 客胜 {{ pc(result?.market_away_win_prob) }}</div>
        <div>盘口：期望 {{ fmtHcap(result?.expected_handicap) }} | 终盘 {{ fmtHcap(result?.market_final_handicap) }} | 判定 {{ result?.handicap_anomaly_label || '—' }}</div>
        <div class="pt-2 border-t">最可能比分：<span class="font-mono">{{ ((result?.likely_scores||[]) as any[]).slice(0,3).map((i:any)=>i.score).join('、') || '—' }}</span></div>
      </div>
      <template #footer>
        <el-button @click="resultDlg=false">关闭</el-button>
        <el-button type="primary" @click="resultDlg=false">好的</el-button>
      </template>
    </el-dialog>

    <!-- 阵容弹窗 -->
    <el-dialog v-model="rosterDlg" :title="rosterSide==='home'?'编辑主队阵容':'编辑客队阵容'" width="420px">
      <el-table :data="rosterList" size="small" border>
        <el-table-column label="球员" width="160"><template #default="{ row }"><el-input v-model="row.name" /></template></el-table-column>
        <el-table-column label="位置"><template #default="{ row }"><el-input v-model="row.position" /></template></el-table-column>
        <el-table-column label="评分"><template #default="{ row }"><el-input-number v-model="row.rating" :step="0.1" :precision="1" :max="10" /></template></el-table-column>
        <el-table-column width="50" align="center">
          <template #default="{ $index }">
            <el-button size="small" type="danger" text @click="removeRosterPlayer($index)">删</el-button>
          </template>
        </el-table-column>
      </el-table>
      <div class="mt-3 text-right">
        <el-button size="small" @click="addRosterPlayer">+ 添加球员</el-button>
      </div>
      <template #footer>
        <el-button @click="rosterDlg=false">取消</el-button>
        <el-button type="primary" @click="saveRoster">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.page {
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 16px;
}
.heatmap-wrap { overflow-x: auto; }
.heatmap { min-width: 800px; height: 380px; }
.scroll-x { overflow-x: auto; }
.pixel-spinner { display: inline-flex; gap: 8px; align-items: center; margin: 8px 0; }
.pixel-spinner .sq { width: 14px; height: 14px; background: var(--pixel-primary); border: 2px solid var(--pixel-border); box-shadow: 4px 4px 0 var(--pixel-shadow); image-rendering: pixelated; }
.pixel-spinner .sq:nth-child(1) { animation: pixel-bounce 600ms steps(4) infinite; }
.pixel-spinner .sq:nth-child(2) { animation: pixel-bounce 600ms steps(4) infinite 100ms; }
.pixel-spinner .sq:nth-child(3) { animation: pixel-bounce 600ms steps(4) infinite 200ms; }
@keyframes pixel-bounce { 0% { transform: translateY(0); } 50% { transform: translateY(-6px); } 100% { transform: translateY(0); } }

.kf-section { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:16px; }
.kf-dist { display:flex; gap:16px; }
.kf-item { flex:1; background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:12px; text-align:center; }
.kf-num { font-size:24px; font-weight:700; color:#111827; }
.kf-label { font-size:12px; color:#6b7280; margin-bottom:8px; }
.kf-bar { height:8px; background:#e5e7eb; border-radius:6px; overflow:hidden; }
.kf-bar span { display:block; height:8px; }
.kf-bar .home { background:#2e7df6; }
.kf-bar .draw { background:#8892a7; }
.kf-bar .away { background:#ff4d4f; }
.kf-top { margin-top:12px; }
.kf-top-title { font-weight:600; font-size:14px; margin-bottom:4px; color:#374151; }
.kf-top-list { display:flex; gap:12px; flex-wrap:wrap; font-size:12px; color:#374151; }
.kf-pairs { display:grid; grid-template-columns: repeat(2,1fr); gap:12px; margin-top:12px; }
.kf-pair { background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
.kf-pair-title { font-size:12px; color:#6b7280; margin-bottom:6px; }
.kf-pair-bars { position:relative; height:8px; background:#e5e7eb; border-radius:6px; overflow:hidden; display:flex; }
.kf-pair-bars .left { background:#2e7df6; }
.kf-pair-bars .right { background:#8892a7; }
.kf-pair-values { display:flex; justify-content:space-between; font-size:12px; margin-top:6px; color:#374151; }

.kf-hcap { display:flex; gap:12px; }
.kf-metric { flex:1; background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
.kf-metric-title { font-size:12px; color:#6b7280; margin-bottom:6px; }
.kf-metric-value { font-size:18px; font-weight:700; color:#111827; }
.kf-chip { display:inline-block; padding:4px 8px; background:#eef2ff; color:#374151; border:1px solid #e5e7eb; border-radius:999px; font-size:12px; }
.kf-bar-wrap { margin-top:12px; }
.kf-bar { height:8px; background:#e5e7eb; border-radius:6px; overflow:hidden; }
.kf-bar > span { display:block; height:8px; }
.sev.success { background:#10b981; }
.sev.warning { background:#f59e0b; }
.sev.danger { background:#ef4444; }
.kf-bar-text { display:flex; justify-content:space-between; font-size:12px; color:#374151; margin-top:6px; }
.kf-upset { display:grid; grid-template-columns: repeat(3,1fr); gap:12px; align-items:center; }
.kf-upset .kf-bar { grid-column: 1 / -1; }
.kf-upset .kf-note { grid-column: 1 / -1; font-size:12px; color:#6b7280; margin-top:4px; }
.kf-ana { background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
.kf-pre { white-space: pre; overflow-x: auto; font-size:13px; color:#111827; }
.kf-pre.small { font-size:12px; }
.kf-match { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
.kf-teams { display:flex; align-items:center; gap:12px; }
.kf-team-name { font-size:18px; font-weight:700; color:#111827; }
.kf-vs { font-size:12px; color:#6b7280; }
.kf-meta { margin-top:4px; font-size:12px; color:#6b7280; display:flex; gap:8px; align-items:center; }
.kf-comp-grid { display:grid; grid-template-columns: 1fr; gap:10px; }
.kf-company { background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:8px; }
.kf-company-name { font-size:13px; font-weight:600; color:#374151; margin-bottom:6px; }
.kf-3bars { display:grid; grid-template-columns: 1fr; gap:6px; }
.kf-3bars .bar { position:relative; height:8px; background:#e5e7eb; border-radius:6px; overflow:hidden; }
.kf-3bars .bar.home { background:#dbeafe; }
.kf-3bars .bar.draw { background:#e5e7eb; }
.kf-3bars .bar.away { background:#fee2e2; }
.kf-3bars .bar > span { position:absolute; right:6px; top:50%; transform: translateY(-50%); font-size:10px; color:#374151; }

.kf-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:16px; }
.kf-card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }
.kf-card-title { font-size:16px; font-weight:700; color:#111827; }
.kf-card-actions { display:flex; gap:8px; }
.kf-subtitle { font-size:14px; font-weight:600; color:#374151; margin-bottom:8px; }
.kf-form { }
.kf-form :deep(.el-form-item__label) { font-size:12px; color:#6b7280; padding-bottom:6px; }
.kf-help { font-size:12px; color:#6b7280; margin-top:4px; }
.kf-odds-list { display:grid; grid-template-columns: 1fr; gap:12px; }
.kf-odds-row { background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
.kf-odds-split { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
.kf-odds-grid { display:grid; grid-template-columns: repeat(3,1fr); gap:8px; }
.kf-odds-grid3 { display:grid; grid-template-columns: repeat(3,1fr); gap:8px; }
.kf-odds-cell { display:flex; flex-direction:column; gap:4px; }
.kf-odds-label { font-size:12px; color:#6b7280; }
.kf-actions { margin-top:16px; display:flex; gap:8px; }
</style>

<script setup lang="ts">
import { ref, reactive, nextTick, watch, computed } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import api, { predict, getTeamMetrics, listCompanies } from '@/api'
import type { TeamMetrics } from '@/api'
type PredictResult = {
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
  backtest_summary?: string
  analysis?: string
  analysis_struct?: { summary?: unknown }
  score_matrix?: number[][]
  calibration_curve?: { pred: number, actual: number }[]
}

// 盘口选项
const handicapOpts = ['-2.5','-2.25','-2','-1.75','-1.5','-1.25','-1','-0.75','-0.5','-0.25','0','+0.25','+0.5','+0.75','+1','+1.25','+1.5','+1.75','+2','+2.25','+2.5']

// 表单结构（与原页面对齐）
const form = reactive({
  home_team: {
    name: '',
    formation: '',
    ranking: 1,
    games_played: 30,
    xg: 2.1,
    xga: 0.9,
    xpts: 2.3,
    season_goals_for: 10,
    season_goals_against: 3,
    npxg: undefined as number|undefined,
    npxga: undefined as number|undefined,
    ppda: undefined as number|undefined,
    oppda: undefined as number|undefined,
    dc: undefined as number|undefined,
    odc: undefined as number|undefined,
    wins: undefined as number|undefined,
    draws: undefined as number|undefined,
    losses: undefined as number|undefined,
    points: undefined as number|undefined,
    xpxgd: undefined as number|undefined,
    roster: [] as {name:string,position:string,rating:number}[]
  },
  away_team: {
    name: '',
    formation: '',
    ranking: 1,
    games_played: 30,
    xg: 1.8,
    xga: 1.1,
    xpts: 1.9,
    season_goals_for: 8,
    season_goals_against: 5,
    npxg: undefined as number|undefined,
    npxga: undefined as number|undefined,
    ppda: undefined as number|undefined,
    oppda: undefined as number|undefined,
    dc: undefined as number|undefined,
    odc: undefined as number|undefined,
    wins: undefined as number|undefined,
    draws: undefined as number|undefined,
    losses: undefined as number|undefined,
    points: undefined as number|undefined,
    xpxgd: undefined as number|undefined,
    roster: [] as {name:string,position:string,rating:number}[]
  },
  league_preset: '' as ''|'英超/德甲'|'意甲/法甲'|'西甲',
  odds: [] as {
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
})

// 阵容弹窗
const rosterDlg = ref(false)
const rosterSide = ref<'home'|'away'>('home')
const rosterList = ref<{name:string,position:string,rating:number}[]>([])
const openRosterModal = (side:'home'|'away') => {
  rosterSide.value = side
  const key = side === 'home' ? 'home_team' : 'away_team'
  rosterList.value = JSON.parse(JSON.stringify(form[key].roster))
  rosterDlg.value = true
}
const addRosterPlayer = () => rosterList.value.push({ name:'', position:'', rating:6.5 })
const removeRosterPlayer = (i:number) => rosterList.value.splice(i,1)
const saveRoster = () => {
  const key = rosterSide.value === 'home' ? 'home_team' : 'away_team'
  form[key].roster = rosterList.value.filter(p=>p.name.trim())
  rosterDlg.value = false
}
const homeRosterSummary = computed(()=> form.home_team.roster.length ? `已设 ${form.home_team.roster.length} 人` : '未设置阵容')
const awayRosterSummary = computed(()=> form.away_team.roster.length ? `已设 ${form.away_team.roster.length} 人` : '未设置阵容')

// 赔率表格
const addOddsRow = () => form.odds.push({
  company:'',
  initial_home_win:0, initial_draw:0, initial_away_win:0,
  initial_handicap:'0', initial_handicap_home_odds:1.00, initial_handicap_away_odds:1.00,
  final_home_win:0, final_draw:0, final_away_win:0,
  final_handicap:'0', final_handicap_home_odds:1.00, final_handicap_away_odds:1.00
})
const removeOddsRow = (i:number) => form.odds.splice(i,1)

// 进度
const loading = ref(false)
const progressVisible = ref(false)
const progressPercent = ref(0)
const progressText = ref('')
const resultDlg = ref(false)
const cancelProgress = () => {
  progressVisible.value = false
  progressPercent.value = 0
  progressText.value = ''
}

// 结果
const result = ref<PredictResult|null>(null)
const aiShow = ref<'text'|'structured'|'json'>('text')
const companyOptions = ref<string[]>(['365','皇冠（crown）','澳门','易胜博'])

// 计算各公司隐含概率
const impliedProbabilities = computed(() => {
  if (!form.odds || form.odds.length === 0) return []
  return form.odds.map(odds => {
    const h = Number(odds.final_home_win || 0)
    const d = Number(odds.final_draw || 0)
    const a = Number(odds.final_away_win || 0)
    if (h > 0 && d > 0 && a > 0) {
      const ph = 1 / h
      const pd = 1 / d
      const pa = 1 / a
      const s = ph + pd + pa
      return {
        company: odds.company || '—',
        homeWin: Math.round((ph / s) * 100),
        draw: Math.round((pd / s) * 100),
        awayWin: Math.round((pa / s) * 100)
      }
    }
    return { company: odds.company || '—', homeWin: '—', draw: '—', awayWin: '—' }
  })
})

const impliedBarData = computed(() => {
  return impliedProbabilities.value.map(it => {
    const home = typeof it.homeWin === 'number' ? it.homeWin : 0
    const draw = typeof it.draw === 'number' ? it.draw : 0
    const away = typeof it.awayWin === 'number' ? it.awayWin : 0
    return { company: it.company, home, draw, away }
  })
})

// 工具函数
const pc = (v:number|undefined) => (v===undefined ? '—' : `${(v*100).toFixed(1)}%`)
const fmtHcap = (v:number|undefined) => (v===undefined ? '—' : (v>0?`+${v.toFixed(2)}`:v.toFixed(2)))
const fmtScore = (v:number|undefined) => (v===undefined ? '—' : `${Math.round((v||0)*100)} / 100`)
 
const upsetTag = computed(()=>{
  const s = result.value?.upset_score
  if(s===undefined) return 'info'
  return s>0.7?'danger':s>0.4?'warning':'success'
})

const hcapSev = computed(()=>{
  const s = result.value?.handicap_anomaly_score
  if(s===undefined) return 'success'
  return s>0.7?'danger':s>0.4?'warning':'success'
})

const pctVal = (v:number|undefined|null) => {
  const x = typeof v === 'number' ? v : 0
  return `${Math.round(x*100)}%`
}

const totals = computed(()=>{
  const mx = result.value?.score_matrix
  if(!mx) return null
  let over15=0,under15=0,over25=0,under25=0,over35=0,under35=0,btts_yes=0,btts_no=0
  for(let i=0;i<mx.length;i++){
    const row = mx[i] || []
    for(let j=0;j<row.length;j++){
      const p = Number(row[j]||0)
      const sum = i + j
      if(sum>=2) over15+=p; else under15+=p
      if(sum>=3) over25+=p; else under25+=p
      if(sum>=4) over35+=p; else under35+=p
      if(i>0 && j>0) btts_yes+=p; else btts_no+=p
    }
  }
  const clamp = (x:number)=> Math.max(0, Math.min(1, x))
  return {
    over15: clamp(over15), under15: clamp(under15),
    over25: clamp(over25), under25: clamp(under25),
    over35: clamp(over35), under35: clamp(under35),
    btts_yes: clamp(btts_yes), btts_no: clamp(btts_no)
  }
})

const topScores = computed(()=> {
  const mx = result.value?.score_matrix
  if(!mx) {
    const lst = (result.value?.likely_scores||[]).slice(0,3)
    return lst.map(s=>({ score: s.score, prob: undefined as number|undefined }))
  }
  const arr: {score:string, prob:number}[] = []
  for(let i=0;i<mx.length;i++){
    const row = mx[i] || []
    for(let j=0;j<row.length;j++){
      const p = Number(row[j]||0)
      arr.push({ score: `${i}-${j}`, prob: p })
    }
  }
  arr.sort((a,b)=>b.prob - a.prob)
  return arr.slice(0,3)
})

// 提交预测
const submitPredict = async () => {
  if (loading.value) return
  if(!form.home_team.name || !form.away_team.name){ ElMessage.warning('请填写球队名称'); return }
  loading.value = true
  progressVisible.value = true
  progressPercent.value = 10
  progressText.value = '正在提交...'
  try{
    // 构造 payload（与原页面对齐）
    if (!form.odds || form.odds.length === 0) { ElMessage.warning('请至少添加一家公司的赔率'); loading.value=false; return }
    const payload = {
      home_team: { ...form.home_team },
      away_team: { ...form.away_team },
      league_preset: form.league_preset || undefined,
      odds_data: form.odds
    }
    progressPercent.value = 30
    progressText.value = '模型计算中...'
    const res = await predict(payload)
    progressPercent.value = 100
    progressText.value = '完成'
    result.value = res
    nextTick(()=>{ renderCharts() })
    progressVisible.value = false
    resultDlg.value = true
  }catch(e){
    const msg = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : ''
    const code = (e && typeof e === 'object' && 'code' in e) ? String((e as {code?: unknown}).code) : ''
    if (code === 'ERR_ABORTED' || code === 'ERR_CANCELED' || msg.includes('ERR_ABORTED')) {
      cancelProgress()
      loading.value = false
      return
    }
    if (code === 'ECONNABORTED' || code === 'ERR_NETWORK') {
      progressText.value = '网络中断/超时，正在重试...'
      try {
        const res2 = await api.post('/predict', {
          home_team: { ...form.home_team },
          away_team: { ...form.away_team },
          league_preset: form.league_preset || undefined,
          odds_data: form.odds
        }, { timeout: 120000 }).then(r=>r.data)
        result.value = res2
        nextTick(()=>{ renderCharts() })
        progressVisible.value = false
        resultDlg.value = true
        return
      } catch (e2) {
        const m2 = (e2 && typeof e2 === 'object' && 'message' in e2) ? String((e2 as {message?: unknown}).message) : '未知错误'
        ElMessage.error('预测失败：'+m2)
      }
    } else {
      const m = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : '未知错误'
      ElMessage.error('预测失败：'+m)
    }
    cancelProgress()
  }finally{
    loading.value = false
  }
}

// 加载模拟数据
const loadMockData = () => {
  Object.assign(form.home_team, {
    name:'曼城', formation:'4-2-3-1', ranking:2, games_played:30, xg:2.1, xga:0.9, xpts:2.3, season_goals_for:68, season_goals_against:22,
    npxg:1.9, npxga:0.8, ppda:8.2, oppda:11.5, dc:2.1, odc:0.8, wins:20, draws:6, losses:4, points:66, xpxgd:1.1
  })
  Object.assign(form.away_team, {
    name:'利物浦', formation:'4-3-3', ranking:3, games_played:30, xg:1.8, xga:1.1, xpts:1.9, season_goals_for:65, season_goals_against:30,
    npxg:1.7, npxga:1.0, ppda:9.1, oppda:10.8, dc:1.8, odc:1.0, wins:18, draws:8, losses:4, points:62, xpxgd:0.7
  })
  form.league_preset = '英超/德甲'
  const names = companyOptions.value.slice(0,4)
  form.odds = [
    { company:names[0]||'365',
      initial_home_win:1.90, initial_draw:3.50, initial_away_win:4.20,
      final_home_win:1.85, final_draw:3.60, final_away_win:4.30,
      initial_handicap:'-0.5', initial_handicap_home_odds:1.90, initial_handicap_away_odds:1.90,
      final_handicap:'-0.5', final_handicap_home_odds:1.88, final_handicap_away_odds:1.92 },
    { company:names[1]||'皇冠（crown）',
      initial_home_win:1.92, initial_draw:3.45, initial_away_win:4.10,
      final_home_win:1.88, final_draw:3.55, final_away_win:4.25,
      initial_handicap:'-0.5', initial_handicap_home_odds:1.87, initial_handicap_away_odds:1.95,
      final_handicap:'-0.5', final_handicap_home_odds:1.85, final_handicap_away_odds:1.98 },
    { company:names[2]||'澳门',
      initial_home_win:1.95, initial_draw:3.60, initial_away_win:3.90,
      final_home_win:1.90, final_draw:3.65, final_away_win:4.05,
      initial_handicap:'-0.5', initial_handicap_home_odds:1.88, initial_handicap_away_odds:1.92,
      final_handicap:'-0.5', final_handicap_home_odds:1.86, final_handicap_away_odds:1.96 },
    { company:names[3]||'易胜博',
      initial_home_win:1.93, initial_draw:3.55, initial_away_win:4.15,
      final_home_win:1.89, final_draw:3.60, final_away_win:4.28,
      initial_handicap:'-0.5', initial_handicap_home_odds:1.89, initial_handicap_away_odds:1.91,
      final_handicap:'-0.5', final_handicap_home_odds:1.87, final_handicap_away_odds:1.97 }
  ]
  ElMessage.success('已加载模拟数据')
}

// 图表渲染
const heatmapDom = ref<HTMLDivElement>()
const calibDom = ref<HTMLDivElement>()
const chartFusedDom = ref<HTMLDivElement>()
let heatmapInst: echarts.ECharts|null = null
let calibInst: echarts.ECharts|null = null
let fusedInst: echarts.ECharts|null = null

const renderCharts = () => {
  if(!result.value) return
  // 热力图
  if(result.value?.score_matrix && heatmapDom.value){
    if(!heatmapInst) heatmapInst = echarts.init(heatmapDom.value)
    const hdata = result.value.score_matrix as number[][]
    const maxG = hdata.length
    heatmapInst.setOption({
      grid:{left:'40',right:'20',bottom:'40',top:'40'},
      xAxis:{type:'category',data:Array.from({length:maxG},(_,i)=>i),splitArea:{show:true}},
      yAxis:{type:'category',data:Array.from({length:maxG},(_,i)=>i),splitArea:{show:true}},
      visualMap:{min:0,max:Math.max(...hdata.flat()),calculable:true,inOrient:'horizontal',left:'center',bottom:'0%',inRange:{color:['#fff','#bd0000']}},
      tooltip:{formatter:(p:{value:number[]})=>`${p.value[1]}-${p.value[0]}: ${(p.value[2]||0).toFixed(3)}`},
      series:[{type:'heatmap',data:hdata.map((row,y)=>row.map((v,x)=>([x,y,v]))).flat(),label:{show:false},emphasis:{itemStyle:{shadowBlur:10,shadowColor:'rgba(0,0,0,0.5)'}}}]
    })
    const cw = Math.max(800, maxG * 60)
    if (heatmapDom.value) {
      heatmapDom.value.style.width = `${cw}px`
    }
    heatmapInst.resize()
  }
  // 校准曲线
  if(result.value?.calibration_curve && calibDom.value){
    if(!calibInst) calibInst = echarts.init(calibDom.value)
    const c = result.value.calibration_curve as {pred:number,actual:number}[]
    calibInst.setOption({
      title:{text:'校准曲线',left:'center'},
      tooltip:{trigger:'axis'},
      xAxis:{name:'预测概率',min:0,max:1},
      yAxis:{name:'实际命中率',min:0,max:1},
      series:[
        {name:'校准',type:'line',data:c.map((p:{pred:number,actual:number})=>[p.pred,p.actual]),smooth:true,symbol:'circle',symbolSize:6},
        {name:'理想',type:'line',data:[[0,0],[1,1]],lineStyle:{type:'dashed',color:'#999'},symbol:'none'}
      ]
    })
  }
  // 三路概率柱状图
  if(chartFusedDom.value){
    if(!fusedInst) fusedInst = echarts.init(chartFusedDom.value)
    fusedInst.setOption({
      title:{text:'三路融合概率',left:'center'},
      tooltip:{trigger:'axis'},
      legend:{data:['模型','市场','融合'],bottom:0},
      xAxis:{type:'category',data:['主胜','平局','客胜']},
      yAxis:{type:'value',name:'概率',axisLabel:{formatter:'{value}%'}},
      series:[
        {name:'模型',type:'bar',data:[result.value.home_win_prob,result.value.draw_prob,result.value.away_win_prob].map(v=>((v||0)*100).toFixed(1))},
        {name:'市场',type:'bar',data:[result.value.market_home_win_prob,result.value.market_draw_prob,result.value.market_away_win_prob].map(v=>((v||0)*100).toFixed(1))},
        {name:'融合',type:'bar',data:[result.value.fused_home_win_prob,result.value.fused_draw_prob,result.value.fused_away_win_prob].map(v=>((v||0)*100).toFixed(1))}
      ]
    })
  }
}

// 自动填充球队数据
const autoFillTeamData = async (side: 'home' | 'away') => {
  try {
    const key = side === 'home' ? 'home_team' : 'away_team'
    const team = form[key]
    if (!team || !team.name) {
      return
    }
    const teamName = team.name.trim()
    if (!teamName) {
      return
    }

    ElMessage.info(`正在获取 ${teamName} 的数据...`)
    
    let metrics: TeamMetrics
    try {
      metrics = await getTeamMetrics(teamName)
      ElMessage.success(`成功获取 ${teamName} 的数据`)
    } catch (apiError) {
      const resp = (apiError && typeof apiError === 'object' && 'response' in apiError) ? (apiError as {response?: {status?: number}}).response : undefined
      if (resp?.status === 404) {
        ElMessage.warning(`未找到 ${teamName} 的数据，请手动输入`)
        return
      }
      throw apiError
    }
    
    if (!form || !form[key] || form[key].name !== teamName) {
      return
    }
    
    if (!metrics || typeof metrics !== 'object') {
      ElMessage.error('返回的数据格式错误')
      return
    }
    
    form[key].games_played = metrics.games_played ?? 0
    form[key].xg = metrics.xg ?? 0
    form[key].xga = metrics.xga ?? 0
    form[key].xpts = metrics.xpts ?? 0
    form[key].season_goals_for = metrics.goals_for ?? 0
    form[key].season_goals_against = metrics.goals_against ?? 0
    form[key].wins = metrics.wins ?? 0
    form[key].draws = metrics.draws ?? 0
    form[key].losses = metrics.losses ?? 0
    form[key].points = metrics.points ?? 0
    if (metrics.npxg !== undefined && metrics.npxg !== null) form[key].npxg = metrics.npxg
    if (metrics.npxga !== undefined && metrics.npxga !== null) form[key].npxga = metrics.npxga
    if (metrics.ppda !== undefined && metrics.ppda !== null) form[key].ppda = metrics.ppda
    if (metrics.oppda !== undefined && metrics.oppda !== null) form[key].oppda = metrics.oppda
    if (metrics.dc !== undefined && metrics.dc !== null) form[key].dc = metrics.dc
    if (metrics.odc !== undefined && metrics.odc !== null) form[key].odc = metrics.odc
    if (metrics.xpxgd !== undefined && metrics.xpxgd !== null) form[key].xpxgd = metrics.xpxgd
    ElMessage.success(`已自动填充 ${teamName} 的数据`)
  } catch (error) {
    const m = (error && typeof error === 'object' && 'message' in error) ? String((error as {message?: unknown}).message) : '未知错误'
    ElMessage.error(`自动填充失败: ${m}`)
  }
}

// 监听球队名称变化
watch(() => form.home_team.name, (newName) => {
  console.log(`主队名称变化: ${newName}`)
  if (newName && newName.trim()) {
    // 延迟执行，避免用户输入过程中频繁请求
    setTimeout(() => {
      // 确保表单数据仍然存在且未被修改
      if (form && form.home_team && form.home_team.name === newName) {
        console.log(`触发主队自动填充: ${newName}`)
        autoFillTeamData('home')
      } else {
        console.log(`主队名称已改变，跳过自动填充`)
      }
    }, 1000)
  } else {
    console.log(`主队名称为空，跳过自动填充`)
  }
})

watch(() => form.away_team.name, (newName) => {
  console.log(`客队名称变化: ${newName}`)
  if (newName && newName.trim()) {
    // 延迟执行，避免用户输入过程中频繁请求
    setTimeout(() => {
      // 确保表单数据仍然存在且未被修改
      if (form && form.away_team && form.away_team.name === newName) {
        console.log(`触发客队自动填充: ${newName}`)
        autoFillTeamData('away')
      } else {
        console.log(`客队名称已改变，跳过自动填充`)
      }
    }, 1000)
  } else {
    console.log(`客队名称为空，跳过自动填充`)
  }
})

// 初始化一条空赔率
addOddsRow()

// 测试函数 - 用于调试
const testAutoFill = async () => {
  console.log('开始测试自动填充功能...')
  try {
    // 直接测试API调用
    console.log('直接测试API调用...')
    const testMetrics = await getTeamMetrics('Napoli')
    console.log('API调用成功:', testMetrics)
    ElMessage.success('API调用成功')
    
    // 设置测试球队名称
    form.home_team.name = 'Napoli'
    console.log('已设置主队名称为 Napoli')
    
    // 等待1秒后触发自动填充
    setTimeout(() => {
      console.log('手动触发自动填充...')
      autoFillTeamData('home')
    }, 1000)
    
  } catch (error) {
    const err = error
    console.error('测试失败:', err)
    const m = (err && typeof err === 'object' && 'message' in err) ? String((err as {message?: unknown}).message) : String(err)
    ElMessage.error('测试失败: '+m)
  }
}

// 页面加载完成后自动运行测试
setTimeout(() => {
  console.log('页面加载完成，开始测试...')
  testAutoFill()
}, 2000)

// 加载公司列表
nextTick(async () => {
  try {
    const list = await listCompanies()
    if (Array.isArray(list.companies) && list.companies.length >= 4) {
      companyOptions.value = list.companies
    }
  } catch {}
})
</script>
