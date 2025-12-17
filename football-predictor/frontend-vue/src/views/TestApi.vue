<template>
  <div class="p-8">
    <h1 class="text-xl font-bold mb-4">API 连通测试</h1>
    <el-button type="primary" @click="checkAIStatus">获取 AI 状态</el-button>
    <el-button type="success" @click="fetchConfig">读取配置</el-button>
    <div v-if="msg" class="mt-4 text-sm text-gray-700">{{ msg }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { getAIStatus, getConfig } from '@/api'

const msg = ref('')

const checkAIStatus = async () => {
  try {
    const res = await getAIStatus()
    msg.value = 'AI 状态：' + JSON.stringify(res)
  } catch (e) {
    const m = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : String(e)
    msg.value = '错误：' + m
  }
}

const fetchConfig = async () => {
  try {
    const res = await getConfig()
    msg.value = '配置：' + JSON.stringify(res)
  } catch (e) {
    const m = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : String(e)
    msg.value = '错误：' + m
  }
}
</script>