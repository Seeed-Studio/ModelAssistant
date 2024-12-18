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
                { text: '快速上手', link: '/zh_cn/introduction/quick_start' },
                { text: '安装指南', link: '/zh_cn/introduction/installation' }
            ]
        },
        {
            text: '基础教程',
            collapsed: false,
            items: [
                {
                    text: '流程概览',
                    link: '/zh_cn/tutorials/overview',
                },
                {
                    text: '模型训练与导出',
                    link: '/zh_cn/tutorials/training/overview',
                    items: [
                        { text: 'FOMO 模型', link: '/zh_cn/tutorials/training/fomo' },
                        { text: 'PFLD 模型', link: '/zh_cn/tutorials/training/pfld' },
                        { text: 'RTMDet 模型', link: '/zh_cn/tutorials/training/rtmdet' },
                        { text: 'VAE 模型', link: '/zh_cn/tutorials/training/vae' }
                    ]
                },
                {
                    text: '模型部署',
                    link: '/zh_cn/tutorials/deploy/overview',
                    items: [
                        { text: 'Grove Vision AI V2', link: '/zh_cn/tutorials/deploy/grove_vision_ai_v2' },
                        { text: 'XIAO ESP32S3 Sense', link: '/zh_cn/tutorials/deploy/xiao_esp32s3' }
                    ]
                }
            ]
        },
        {
            text: '自定义',
            collapsed: false,
            items: [
                { text: '基础配置结构', link: '/zh_cn/custom/basics' },
                { text: '模型结构', link: '/zh_cn/custom/model' },
                { text: '训练与验证管线', link: '/zh_cn/custom/pipelines' },
                { text: '优化器', link: '/zh_cn/custom/optimizer' },
            ]
        },
        {
            text: '数据集',
            collapsed: false,
            items: [
                { text: '公共数据集', link: '/zh_cn/datasets/public' },
                { text: '自制数据集', link: '/zh_cn/datasets/custom' },
                { text: '数据集格式与扩展', link: '/zh_cn/datasets/extension' },
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
