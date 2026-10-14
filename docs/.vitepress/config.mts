import { defineConfig } from 'vitepress'

// 注意：base 必须写成 '/<仓库名>/'，否则 GitHub Pages 上的资源会 404。
// 如果之后把仓库改名，这里要同步改。
export default defineConfig({
  title: 'GPU 笔记',
  description: 'hello-gpu 组队学习笔记',
  base: '/hello-gpu-notes/',
  lang: 'zh-CN',

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'hello-gpu', link: '/notes/task0-environment-setup' },
      { text: '官方教程', link: 'https://datawhalechina.github.io/hello-gpu/' }
    ],

    sidebar: [
      {
        text: 'hello-gpu 组队学习',
        items: [
          { text: 'Task 0 · 环境准备（第 1 章）', link: '/notes/task0-environment-setup' }
        ]
      }
    ],

    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '回到顶部',
    darkModeSwitchLabel: '深色模式',
    sidebarMenuLabel: '目录',
    search: { provider: 'local' }
  }
})
