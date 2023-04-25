import { defineConfig } from 'vitepress'

import en_US from './locales/en_US'
import zh_CN from './locales/zh_CN'

export default defineConfig({
    head: [
        ['link', { rel: 'icon', type: 'image/png', href: 'favicon.png' }],
        ['meta', { property: 'og:type', content: 'website' }],
        ['meta', { property: 'og:title', content: 'EdgeLab' }],
        ['meta', { property: 'og:image', content: 'og-image.png' }],
        ['meta', { property: 'og:url', content: 'https://github.com/Seeed-Studio/EdgeLab' }],
        ['meta', { property: 'og:description', content: 'Seeed Studio EdgeLab is an open-source project focused on embedded AI.' }],
        ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
        ['meta', { name: 'twitter:site', content: '@seeedstudio' }],
        ['meta', { name: 'theme-color', content: '#051726' }]
    ],

    base: '/EdgeLab',
    title: 'EdgeLab',
    lastUpdated: true,
    cleanUrls: true,

    locales: {
        root: {
            label: 'English',
            lang: en_US.lang,
            description: en_US.description,
            themeConfig: en_US.themeConfig
        },
        zh_cn: {
            label: '简体中文',
            lang: zh_CN.lang,
            description: zh_CN.description,
            themeConfig: zh_CN.themeConfig
        }
    },

    themeConfig: {
        search: { provider: 'local' },

        i18nRouting: true,

        socialLinks: [
            { icon: 'github', link: 'https://github.com/Seeed-Studio/Edgelab' }
        ]
    }
})
