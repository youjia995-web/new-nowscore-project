import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) return savedPosition
    return { left: 0, top: 0 }
  },
  routes: [
    {
      path: '/',
      name: 'predict',
      meta: { title: '比赛预测' },
      component: () => import('../views/PredictView.vue'),
    },
    {
      path: '/history',
      name: 'history',
      meta: { title: '预测记录' },
      component: () => import('../views/HistoryView.vue'),
    },
    {
      path: '/settings',
      name: 'settings',
      meta: { title: '系统设置' },
      component: () => import('../views/SettingsView.vue'),
    },
    {
      path: '/reports',
      name: 'reports',
      meta: { title: '回测报告' },
      component: () => import('../views/ReportsView.vue'),
    },
    {
      path: '/test',
      name: 'test',
      meta: { title: 'API 连通测试' },
      component: () => import('../views/TestApi.vue'),
    },
    {
      path: '/:pathMatch(.*)*',
      redirect: '/',
    },
  ],
})

router.afterEach((to) => {
  const base = '霍金AI 足球预测智能体项目'
  const page = (to.meta && (to.meta as { title?: string }).title) || ''
  document.title = page ? `${base} - ${page}` : base
})

export default router
