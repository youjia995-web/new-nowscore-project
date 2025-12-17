<template>
  <div class="p-6 max-w-4xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">系统设置</h1>
    <p class="text-gray-600 mb-6">调整外部 AI 与 GRU 模型的在线开关</p>

    <!-- 运行配置 -->
    <el-card class="mb-6">
      <template #header><div class="font-semibold">运行配置</div></template>
      <el-form :model="runtimeConfig" label-width="160px">
        <el-form-item label="启用外部 AI">
          <el-switch v-model="runtimeConfig.require_ai_api" />
        </el-form-item>
        <el-form-item label="GRU 启用联赛集合">
          <el-input v-model="runtimeConfig.gru_enabled_leagues" placeholder="意甲,serie a,serie_ligue" />
        </el-form-item>
        <el-form-item label="GRU 融合权重">
          <el-slider v-model="runtimeConfig.gru_model_weight" :min="0" :max="1" :step="0.01" show-input />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="saveRuntimeConfig" :loading="saving">保存</el-button>
          <el-button @click="loadRuntimeConfig">刷新</el-button>
          <span class="ml-3 text-sm text-gray-500">{{ statusText }}</span>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 配置列表 - 隐藏，因为后端只提供运行时配置 -->
    <!--
    <el-card>
      <template #header><div class="font-semibold">配置列表</div></template>
      <el-table :data="configs" stripe style="width: 100%">
        <el-table-column prop="key" label="配置项" width="220" />
        <el-table-column prop="value" label="值" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button type="text" @click="openEdit(row)">编辑</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dlg" title="编辑配置" width="400px">
      <el-form :model="editForm" label-width="80px">
        <el-form-item label="配置项">
          <el-input v-model="editForm.key" disabled />
        </el-form-item>
        <el-form-item label="值">
          <el-input v-model="editForm.value" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dlg = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="save">保存</el-button>
      </template>
    </el-dialog>
    -->
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getConfig, setConfig } from '@/api'
import type { ConfigItem } from '@/api'

const configs = ref<ConfigItem[]>([])
// const dlg = ref(false)
// const editForm = ref<ConfigItem>({ key: '', value: '' })
const saving = ref(false)
const statusText = ref('')

// 运行时配置
const runtimeConfig = ref({
  require_ai_api: false,
  gru_enabled_leagues: '',
  gru_model_weight: 0.5
})

const load = async () => {
  try {
    // 由于后端只返回运行时配置对象，不返回配置列表，清空配置列表
    configs.value = []
  } catch (e) {
    const m = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : String(e)
    ElMessage.error('加载配置失败：' + m)
  }
}

const loadRuntimeConfig = async () => {
  try {
    statusText.value = '正在加载...'
    const configData = await getConfig()
    
    // 直接使用类型化的配置数据
    runtimeConfig.value.require_ai_api = configData.require_ai_api
    runtimeConfig.value.gru_enabled_leagues = configData.gru_enabled_leagues
    runtimeConfig.value.gru_model_weight = configData.gru_model_weight
    
    statusText.value = '已加载当前配置'
  } catch (e) {
    statusText.value = '配置读取失败'
    const m = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : String(e)
    ElMessage.error('加载运行时配置失败：' + m)
  }
}

const saveRuntimeConfig = async () => {
  saving.value = true
  try {
    // 直接发送对象格式的配置数据
    const payload = {
      require_ai_api: runtimeConfig.value.require_ai_api,
      gru_enabled_leagues: runtimeConfig.value.gru_enabled_leagues,
      gru_model_weight: runtimeConfig.value.gru_model_weight
    }
    
    await setConfig(payload)
    ElMessage.success('已保存')
    statusText.value = '配置已保存'
  } catch (e) {
    statusText.value = '配置保存失败'
    const m = (e && typeof e === 'object' && 'message' in e) ? String((e as {message?: unknown}).message) : String(e)
    ElMessage.error('保存失败：' + m)
  } finally {
    saving.value = false
  }
}

// const openEdit = (row: ConfigItem) => {
//   editForm.value = { ...row }
//   dlg.value = true
// }

// const save = async () => {
//   saving.value = true
//   try {
//     await setConfig([editForm.value])
//     ElMessage.success('已保存')
//     dlg.value = false
//     await load()
//   } catch (e: any) {
//     ElMessage.error('保存失败：' + e.message)
//   } finally {
//     saving.value = false
//   }
// }

onMounted(() => {
  load()
  loadRuntimeConfig()
})
</script>