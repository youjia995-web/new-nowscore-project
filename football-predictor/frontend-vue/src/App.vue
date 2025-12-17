<script setup lang="ts">
import { RouterView, useRouter } from 'vue-router'

const router = useRouter()
const menus = [
  { index: '/', label: '比赛预测' },
  { index: '/history', label: '预测记录' },
  { index: '/settings', label: '系统设置' },
  { index: '@ext:/legacy/import.html', label: '数据导入' },
  { index: '/reports', label: '回测报告' },
]
const handleNav = (idx: string) => {
  if (idx.startsWith('@ext:')) {
    const url = idx.slice(5)
    window.open(url, '_blank')
    return
  }
  router.push(idx)
}
</script>

<template>
  <el-container class="h-screen">
    <el-header class="leading-none px-0">
      <div class="app-header">
        <div class="app-title">霍金AI 足球预测智能体项目</div>
        <el-menu mode="horizontal" :default-active="$route.path" @select="handleNav">
          <el-menu-item v-for="m in menus" :key="m.index" :index="m.index">{{ m.label }}</el-menu-item>
        </el-menu>
      </div>
    </el-header>
  <el-main>
      <transition name="pixel-dissolve" mode="out-in">
        <RouterView />
      </transition>
  </el-main>
  </el-container>
</template>

<style scoped>
.el-header {
  --el-header-height: 48px;
}
.app-header { display: flex; justify-content: space-between; align-items: center; }
.app-title { font-size: 18px; font-weight: 600; color: #303133; }
.pixel-dissolve-enter-from { opacity: 0; transform: translateY(8px); }
.pixel-dissolve-enter-active { transition: all 240ms steps(6); }
.pixel-dissolve-leave-to { opacity: 0; transform: translateY(-8px); }
.pixel-dissolve-leave-active { transition: all 180ms steps(5); }
</style>
