<template>
  <div class="p-6 max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">预测记录</h1>

    <el-card class="mb-6">
      <template #header>
        <div class="flex justify-between items-center">
          <div class="font-semibold">选择日期</div>
          <div class="flex gap-2">
            <el-button size="small" @click="goToday">今日</el-button>
          </div>
        </div>
      </template>

      <el-calendar v-model="calDate">
        <template #header>
          <div class="flex justify-between items-center">
            <div class="text-sm text-gray-600">当前月份：{{ monthStr(calDate) }}</div>
            <div class="flex gap-2">
              <el-button size="small" @click="prevMonth">上一月</el-button>
              <el-button size="small" @click="nextMonth">下一月</el-button>
            </div>
          </div>
        </template>
        <template #date-cell="{ data }">
          <div
            class="cal-cell"
            :class="{ selected: selectedDate === data.day, faded: data.type !== 'current-month' }"
            @click="onPickDay(data.day)"
          >
            <div class="day">{{ data.day.split('-').slice(2).join('-') }}</div>
            <div v-if="dayCounts[data.day]" class="badge">{{ dayCounts[data.day] }}</div>
          </div>
        </template>
      </el-calendar>
      <div class="text-xs text-gray-500 mt-2">点击某一天查看当天的预测记录；角标表示当天记录数量。</div>
    </el-card>

    <el-card v-if="selectedDate" class="mb-4">
      <template #header><div class="font-semibold">当天记录：{{ selectedDate }}</div></template>
      <div v-if="loading" class="text-sm text-gray-500">正在加载...</div>
      <div v-else-if="records.length === 0" class="text-sm text-gray-500">当天无记录</div>
      <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div v-for="it in records" :key="it.id" class="kf-card">
          <div class="kf-card-header">
            <div class="kf-card-title">{{ it.home_team }} vs {{ it.away_team }}</div>
            <div class="kf-card-actions">
              <el-button size="small" @click="viewDetail(it.id)">查看详情</el-button>
            </div>
          </div>
          <div class="text-xs text-gray-500 mb-2">提交时间：{{ it.created_at }}</div>
          <div class="kf-metrics">
            <div class="kf-metric">
              <div class="kf-metric-title">预测结果</div>
              <div class="kf-metric-value">{{ labelMap[it.pred_outcome] || it.pred_outcome }}（{{ it.pred_confidence }}）</div>
            </div>
            <div class="kf-metric">
              <div class="kf-metric-title">模型峰值</div>
              <div class="kf-metric-value">{{ pct(Math.max(it.pred_probs.home, it.pred_probs.draw, it.pred_probs.away)) }}</div>
            </div>
            <div class="kf-metric">
              <div class="kf-metric-title">市场峰值</div>
              <div class="kf-metric-value">{{ pct(Math.max(it.market_probs.home, it.market_probs.draw, it.market_probs.away)) }}</div>
            </div>
            <div class="kf-metric">
              <div class="kf-metric-title">可能比分</div>
              <div class="kf-metric-value">{{ it.likely_score || '—' }}</div>
            </div>
            <div class="kf-metric">
              <div class="kf-metric-title">实际赛果</div>
              <div class="kf-metric-value">{{ actualText(it.actual) }}</div>
            </div>
          </div>
        </div>
      </div>
    </el-card>

    <el-dialog v-model="detailVisible" title="预测详情" width="60%">
      <div v-if="activeRecord" class="mb-3 text-sm text-gray-600">
        <span class="font-semibold">{{ activeRecord.home_team }} vs {{ activeRecord.away_team }}</span>
        <span class="ml-2">提交时间：{{ activeRecord.created_at }}</span>
      </div>

      <el-card class="mb-3">
        <template #header><div class="font-semibold">AI最终结论</div></template>
        <div v-if="detailJson">
          <div class="mb-1">
            <span class="mr-2">结论：</span>
            <span class="font-semibold">
              {{ finalOutcome()?.label || (labelMap[activeRecord?.pred_outcome||''] || activeRecord?.pred_outcome) }}
            </span>
            <span v-if="finalOutcome()" class="ml-2 text-gray-500">（{{ pct(finalOutcome()!.prob) }}）</span>
          </div>
          <div v-if="(detailJson.analysis_struct && detailJson.analysis_struct.summary && detailJson.analysis_struct.summary.overview)" class="text-sm text-gray-700">
            {{ detailJson.analysis_struct!.summary!.overview }}
          </div>
          <div class="text-sm text-gray-500 mt-2">
            可能比分：{{ (detailJson.likely_scores||[]).slice(0,3).map(s=>s.score).join('，') || (activeRecord?.likely_score || '—') }}
          </div>
        </div>
        <div v-else class="text-sm text-gray-500">未找到详情</div>
      </el-card>

      <el-card class="mb-3">
        <template #header><div class="font-semibold">真实赛果</div></template>
        <div v-if="activeRecord?.actual">
          <div class="text-sm">比分：{{ activeRecord.actual.home_goals }} - {{ activeRecord.actual.away_goals }}（{{ actualText(activeRecord.actual) }}）</div>
        </div>
        <div v-else class="flex items-center gap-2">
          <el-input-number v-model="actualHome" :min="0" :max="20" size="small" :step="1" />
          <span>-</span>
          <el-input-number v-model="actualAway" :min="0" :max="20" size="small" :step="1" />
          <el-button size="small" type="primary" :disabled="actualHome==null || actualAway==null" @click="saveActual">保存赛果</el-button>
          <span class="text-xs text-gray-500">（无自动数据源时支持手动录入）</span>
        </div>
      </el-card>

      <el-divider>调试</el-divider>
      <el-switch v-model="showJson" active-text="显示原始JSON" />
      <pre v-if="showJson && detailJson" class="kf-pre">{{ JSON.stringify(detailJson, null, 2) }}</pre>

      <template #footer>
        <el-button @click="detailVisible=false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
  
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { listPredictionsByDate, getCalendarCounts, getPrediction, updateActualResult, autoFillActualResult, type PredictionLog } from '@/api'

const calDate = ref<Date>(new Date())
const selectedDate = ref<string>('')
const dayCounts = ref<Record<string, number>>({})
const records = ref<PredictionLog[]>([])
const loading = ref(false)
const detailVisible = ref(false)
interface PredictionDetail {
  fused_home_win_prob?: number
  fused_draw_prob?: number
  fused_away_win_prob?: number
  likely_scores?: { score: string }[]
  analysis?: string
  analysis_struct?: { summary?: { overview?: string, fused_probs?: { home_win: number, draw: number, away_win: number } } }
}
const detailJson = ref<PredictionDetail|null>(null)
const activeRecord = ref<PredictionLog|null>(null)
const actualHome = ref<number|null>(null)
const actualAway = ref<number|null>(null)
const showJson = ref(false)

const labelMap: Record<string,string> = { home: '主胜', draw: '平局', away: '客胜' }
const pct = (v:number|null|undefined) => {
  const x = typeof v === 'number' ? v : 0
  return `${(x*100).toFixed(1)}%`
}
const actualText = (a: PredictionLog['actual']) => {
  if (!a) return '—'
  const m = a.outcome === 'H' ? '主胜' : (a.outcome === 'A' ? '客胜' : '平局')
  return `${a.home_goals}-${a.away_goals}（${m}）`
}
const fmtDate = (d: Date) => {
  const y = d.getFullYear()
  const m = String(d.getMonth()+1).padStart(2,'0')
  const dd = String(d.getDate()).padStart(2,'0')
  return `${y}-${m}-${dd}`
}
const monthStr = (d: Date) => {
  const y = d.getFullYear()
  const m = String(d.getMonth()+1).padStart(2,'0')
  return `${y}-${m}`
}

const refreshCalendarCounts = async () => {
  const m = monthStr(calDate.value)
  const res = await getCalendarCounts(m)
  const map: Record<string, number> = {}
  for (const it of res) map[it.date] = it.count
  dayCounts.value = map
}
const loadRecords = async (day: string) => {
  loading.value = true
  try {
    records.value = await listPredictionsByDate(day)
  } finally {
    loading.value = false
  }
}
const onPickDay = (day: string) => {
  selectedDate.value = day
  loadRecords(day)
}
const goToday = () => {
  calDate.value = new Date()
  const day = fmtDate(calDate.value)
  selectedDate.value = day
  refreshCalendarCounts()
  loadRecords(day)
}
const prevMonth = () => {
  const d = new Date(calDate.value)
  d.setMonth(d.getMonth()-1)
  calDate.value = d
}
const nextMonth = () => {
  const d = new Date(calDate.value)
  d.setMonth(d.getMonth()+1)
  calDate.value = d
}
watch(calDate, () => { refreshCalendarCounts() })

const finalOutcome = () => {
  const dj = detailJson.value
  if (!dj) return null
  const fp = dj.analysis_struct?.summary?.fused_probs
  const entries = [
    { k: 'home', v: typeof (dj.fused_home_win_prob ?? fp?.home_win) === 'number' ? (dj.fused_home_win_prob ?? fp!.home_win) as number : 0 },
    { k: 'draw', v: typeof (dj.fused_draw_prob ?? fp?.draw) === 'number' ? (dj.fused_draw_prob ?? fp!.draw) as number : 0 },
    { k: 'away', v: typeof (dj.fused_away_win_prob ?? fp?.away_win) === 'number' ? (dj.fused_away_win_prob ?? fp!.away_win) as number : 0 },
  ]
  const sorted = entries.sort((a,b)=>b.v-a.v)
  const best = sorted.length ? sorted[0] : undefined
  const map: Record<string,string> = { home: '主胜', draw: '平局', away: '客胜' }
  return (best && best.v > 0) ? { label: map[best.k], prob: best.v } : null
}

const viewDetail = async (id: number, rec?: PredictionLog) => {
  detailVisible.value = true
  activeRecord.value = rec || (records.value.find(r => r.id === id) || null)
  const raw = await getPrediction(id) as PredictionDetail
  detailJson.value = raw || null
  if (!activeRecord.value?.actual) {
    try {
      const auto = await autoFillActualResult(id)
      if (auto && auto.actual_outcome) {
        // 刷新当天记录以同步赛果
        await loadRecords(selectedDate.value)
        const upd = records.value.find(r => r.id === id)
        activeRecord.value = upd || activeRecord.value
      }
    } catch {}
    actualHome.value = null
    actualAway.value = null
  } else {
    actualHome.value = activeRecord.value.actual.home_goals
    actualAway.value = activeRecord.value.actual.away_goals
  }
}

const saveActual = async () => {
  if (!activeRecord.value || actualHome.value == null || actualAway.value == null) return
  const r = await updateActualResult(activeRecord.value.id, Number(actualHome.value), Number(actualAway.value))
  activeRecord.value.actual = { home_goals: Number(actualHome.value), away_goals: Number(actualAway.value), outcome: r.actual_outcome }
  const idx = records.value.findIndex(x => x.id === activeRecord.value!.id)
  if (idx >= 0 && records.value[idx]) records.value[idx].actual = activeRecord.value.actual
}

onMounted(() => {
  goToday()
})
</script>

<style scoped>
.cal-cell { position: relative; padding: 8px; cursor: pointer; border-radius: 6px; }
.cal-cell:hover { background: #f3f4f6; }
.cal-cell.selected { outline: 2px solid #3b82f6; }
.cal-cell.faded { color: #9ca3af; }
.cal-cell .day { font-weight: 600; }
.cal-cell .badge { position: absolute; right: 6px; top: 6px; background: #111827; color: #fff; border-radius: 10px; padding: 0 6px; font-size: 12px; line-height: 18px; }
.kf-card { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
.kf-card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.kf-card-title { font-weight:700; color:#111827; }
.kf-card-actions { display:flex; gap:8px; }
.kf-metrics { display:grid; grid-template-columns: repeat(2,1fr); gap:12px; }
.kf-metric-title { font-size:12px; color:#6b7280; }
.kf-metric-value { font-size:14px; color:#111827; font-weight:600; }
.kf-pre { background:#0b1021; color:#b5c0ff; padding:12px; border-radius:8px; overflow:auto; max-height:60vh; }
</style>
