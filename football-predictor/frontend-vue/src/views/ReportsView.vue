<template>
  <div class="p-6 max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">回测报告</h1>
    <el-card class="mb-6">
      <template #header><div class="font-semibold">总体指标</div></template>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <div class="text-sm text-gray-500">样本数</div>
          <div class="text-xl">{{ summary?.samples ?? '—' }}</div>
        </div>
        <div>
          <div class="text-sm text-gray-500">准确率</div>
          <div class="text-xl">{{ fmt(summary?.accuracy) }}</div>
        </div>
        <div>
          <div class="text-sm text-gray-500">ECE</div>
          <div class="text-xl">{{ fmt(summary?.ece) }}</div>
        </div>
        <div>
          <div class="text-sm text-gray-500">平均Brier</div>
          <div class="text-xl">{{ fmt(summary?.avg_brier) }}</div>
        </div>
        <div>
          <div class="text-sm text-gray-500">平均对数损失</div>
          <div class="text-xl">{{ fmt(summary?.avg_logloss) }}</div>
        </div>
      </div>
    </el-card>
    <el-card class="mb-6">
      <template #header><div class="font-semibold">可靠性曲线</div></template>
      <div class="chart" ref="relChartRef"></div>
      <div class="kf-pair-grid">
        <div v-for="b in (summary?.reliability||[])" :key="b.bin" class="kf-pair-row">
          <div class="kf-pair-bin">{{ b.bin }}</div>
          <div class="kf-pair-bars">
            <span class="left" :style="{width: pctVal(b.mean_prob)}"></span>
            <span class="right" :style="{width: pctVal(b.accuracy)}"></span>
          </div>
          <div class="kf-pair-values"><span>预测 {{ fmt(b.mean_prob) }}</span><span>实际 {{ fmt(b.accuracy) }}</span></div>
        </div>
      </div>
    </el-card>
    <el-card>
      <template #header><div class="font-semibold">按类别的校准</div></template>
      <div class="chart" ref="clsChartRef"></div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { getBacktestSummary, getBacktestCalibration, type BacktestSummary, type BacktestCalibration } from '@/api'

const summary = ref<BacktestSummary|null>(null)
const calib = ref<BacktestCalibration|null>(null)
const relChartRef = ref<HTMLDivElement|null>(null)
const clsChartRef = ref<HTMLDivElement|null>(null)

const fmt = (v: number|null|undefined) => {
  if (v === null || v === undefined) return '—'
  if (v <= 1 && v >= 0) return (v*100).toFixed(2) + '%'
  return String(v.toFixed ? v.toFixed(4) : v)
}
 
const pctVal = (v: number|null|undefined) => {
  const x = typeof v === 'number' ? v : 0
  return `${Math.round(x*100)}%`
}

const renderRelChart = () => {
  if (!relChartRef.value || !summary.value) return
  const bins = (summary.value.reliability || []).map(b => b.bin)
  const mean = (summary.value.reliability || []).map(b => (b.mean_prob ?? null))
  const acc = (summary.value.reliability || []).map(b => (b.accuracy ?? null))
  const chart = echarts.init(relChartRef.value)
  chart.setOption({
    grid: { left: 40, right: 20, top: 30, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: bins },
    yAxis: { type: 'value', min: 0, max: 1 },
    legend: { data: ['平均预测', '实际命中'] },
    series: [
      { name: '平均预测', type: 'line', data: mean },
      { name: '实际命中', type: 'line', data: acc },
    ]
  })
}

const renderClsChart = () => {
  if (!clsChartRef.value || !calib.value) return
  const bins = (calib.value.reliability_by_class.home || []).map(b => b.bin)
  const home = (calib.value.reliability_by_class.home || []).map(b => (b.accuracy ?? null))
  const draw = (calib.value.reliability_by_class.draw || []).map(b => (b.accuracy ?? null))
  const away = (calib.value.reliability_by_class.away || []).map(b => (b.accuracy ?? null))
  const chart = echarts.init(clsChartRef.value)
  chart.setOption({
    grid: { left: 40, right: 20, top: 30, bottom: 40 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: bins },
    yAxis: { type: 'value', min: 0, max: 1 },
    legend: { data: ['主胜', '平局', '客胜'] },
    series: [
      { name: '主胜', type: 'line', data: home },
      { name: '平局', type: 'line', data: draw },
      { name: '客胜', type: 'line', data: away },
    ]
  })
}

const loadData = async () => {
  const s = await getBacktestSummary()
  summary.value = s
  const c = await getBacktestCalibration()
  calib.value = c
  nextTick(() => { renderRelChart(); renderClsChart() })
}

onMounted(() => { loadData() })
</script>

<style scoped>
.chart { width: 100%; height: 320px; }
.kf-pair-grid { display:grid; grid-template-columns: 1fr; gap:8px; margin-top:12px; }
.kf-pair-row { background:#f7f9fc; border:1px solid #e5e7eb; border-radius:8px; padding:8px; }
.kf-pair-bin { font-size:12px; color:#6b7280; margin-bottom:4px; }
.kf-pair-bars { position:relative; height:8px; background:#e5e7eb; border-radius:6px; overflow:hidden; display:flex; }
.kf-pair-bars .left { background:#2e7df6; }
.kf-pair-bars .right { background:#ff4d4f; }
.kf-pair-values { display:flex; justify-content:space-between; font-size:12px; margin-top:6px; color:#374151; }
</style>
