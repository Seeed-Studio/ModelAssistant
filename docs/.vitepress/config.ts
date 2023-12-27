import { defineConfig } from 'vitepress'

import en_US from './locales/en_US'
import zh_CN from './locales/zh_CN'

export default defineConfig({
    base: '/',
    title: 'SSCMA',
    lastUpdated: true,
    cleanUrls: true,

    head: [
        ['link', { rel: 'icon', type: 'image/png', href: '/favicon.png' }],
        ['meta', { property: 'og:type', content: 'website' }],
        ['meta', { property: 'og:title', content: 'SSCMA' }],
        ['meta', { property: 'og:image', content: '/og-image.png' }],
        ['meta', { property: 'og:url', content: 'https://github.com/Seeed-Studio/SSCMA' }],
        ['meta', { property: 'og:description', content: 'Seeed SenSeCraft Model Assiant is an open-source project focused on embedded AI.' }],
        ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
        ['meta', { name: 'twitter:site', content: '@seeedstudio' }],
        ['meta', { name: 'theme-color', content: '#051726' }]
    ],

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
        outline: [2, 3],

        socialLinks: [
            { icon: 'github', link: 'https://github.com/Seeed-Studio/SSCMA' }
        ]
    }
})
