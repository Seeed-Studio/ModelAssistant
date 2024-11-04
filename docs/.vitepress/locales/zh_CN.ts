import { defineConfig } from 'vitepress'

export default defineConfig({
    lang: 'zh-CN',
    description: '*SSCMA* 是一个专注于嵌入式人工智能的开源项目。',

    themeConfig: {
        nav: nav(),
        sidebar: { '/zh_cn': sidebar() },

        darkModeSwitchLabel: '外观切换',
        outlineTitle: '页面大纲',
        lastUpdatedText: '最后更新于',
        returnToTopLabel: '返回顶部',

        editLink: {
            pattern: 'https://github.com/Seeed-Studio/ModelAssistant/edit/main/docs/zh_cn/:path',
            text: '对此页面提出建议'
        },

        footer: {
            message: '在 Apache 2.0 授权下发布',
            copyright: '版权所有 © 2023-目前 Seeed Studio 与 SSCMA 贡献者'
        }
    }
})

function nav() {
    return [
        { text: '主页', link: '/zh_cn/' },
        { text: '文档', link: '/zh_cn/introduction/overview' }
    ]
}

function sidebar() {
    return [
        {
            text: '入门指南',
            collapsed: false,
            items: [
                { text: '什么是 SSCMA?', link: '/zh_cn/introduction/overview' },
                { text: '快速上手', link: '/zh_cn/introduction/quickstart' },
            ]
        },
        {
            text: 'Edge Impulse',
            collapsed: false,
            items: [
                {
                    text: 'Edge Impulse 机器学习块',
                    link: '/zh_cn/edgeimpulse/ei_ml_blocks',
                },
            ]
        },
        {
            text: '社区建设',
            collapsed: false,
            items: [
                { text: 'FAQs', link: '/zh_cn/community/faqs' },
                { text: '参考文档', link: '/zh_cn/community/reference' },
                { text: '贡献指南', link: '/zh_cn/community/contributing' },
                { text: '开源许可', link: '/zh_cn/community/license' }
            ]
        }
    ]
}
