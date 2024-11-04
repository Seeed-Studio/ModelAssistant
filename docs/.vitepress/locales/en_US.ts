import { defineConfig } from 'vitepress'

export default defineConfig({
    lang: 'en-US',
    description: 'SSCMA is an open-source project focused on embedded AI.',

    themeConfig: {
        nav: nav(),
        sidebar: { '/': sidebar() },

        editLink: {
            pattern: 'https://github.com/Seeed-Studio/ModelAssistant/edit/main/docs/:path',
            text: 'Suggest changes to this page'
        },

        footer: {
            message: 'Released under the Apache 2.0 License',
            copyright: 'Copyright © 2023-Present Seeed Studio & SSCMA Contributors',
        }
    }
})

function nav() {
    return [
        { text: 'Home', link: '/' },
        { text: 'Documentation', link: '/introduction/overview' }
    ]
}

function sidebar() {
    return [
        {
            text: 'Introduction',
            collapsed: false,
            items: [
                { text: 'What is SSCMA?', link: '/en/introduction/overview' },
                { text: 'Quick Start', link: '/en/introduction/quick_start' },
            ]
        },

        {
            text: 'Edge Impulse',
            collapsed: false,
            items: [
                {
                    text: 'Machine Learning Blocks',
                    link: '/en/edgeimpulse/ei_ml_blocks',
                },
            ]
        },
        {
            text: 'Community',
            collapsed: false,
            items: [
                { text: 'FAQs', link: '/en/community/faqs' },
                { text: 'Reference', link: '/en/community/reference' },
                { text: 'Contribution', link: '/en/community/contributing' },
                { text: 'Copyrights and Licenses', link: '/en/community/license' }
            ]
        }
    ]
}
