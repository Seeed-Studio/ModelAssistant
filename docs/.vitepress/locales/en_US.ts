import { defineConfig } from 'vitepress'

export default defineConfig({
    lang: 'en-US',
    description: '*SSCMA* is an open-source project focused on embedded artificial intelligence.',

    themeConfig: {
        nav: nav(),
        sidebar: { '/': sidebar() },

        darkModeSwitchLabel: 'Toggle Appearance',
        outlineTitle: 'Page Outline',
        lastUpdatedText: 'Last Updated',
        returnToTopLabel: 'Back to Top',

        editLink: {
            pattern: 'https://github.com/Seeed-Studio/ModelAssistant/edit/main/docs/:path',
            text: 'Suggest edits for this page'
        },

        footer: {
            message: 'Published under Apache 2.0 License',
            copyright: 'Copyright © 2023-Present Seeed Studio and SSCMA Contributors'
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
            text: 'Getting Started',
            collapsed: false,
            items: [
                { text: 'What is SSCMA?', link: '/introduction/overview' },
                { text: 'Quick Start', link: '/introduction/quick_start' },
                { text: 'Installation Guide', link: '/introduction/installation' }
            ]
        },
        {
            text: 'Tutorials',
            collapsed: false,
            items: [
                {
                    text: 'Workflow Overview',
                    link: '/tutorials/overview',
                },
                {
                    text: 'Model Training and Export',
                    link: '/tutorials/training/overview',
                    items: [
                        { text: 'FOMO Model', link: '/tutorials/training/fomo' },
                        { text: 'PFLD Model', link: '/tutorials/training/pfld' },
                        { text: 'RTMDet Model', link: '/tutorials/training/rtmdet' },
                        { text: 'VAE Model', link: '/tutorials/training/vae' }
                    ]
                },
                {
                    text: 'Model Deployment',
                    link: '/tutorials/deploy/overview',
                    items: [
                        { text: 'Grove Vision AI V2', link: '/tutorials/deploy/grove_vision_ai_v2' },
                        { text: 'XIAO ESP32S3 Sense', link: '/tutorials/deploy/xiao_esp32s3' }
                    ]
                }
            ]
        },
        {
            text: 'Customization',
            collapsed: false,
            items: [
                { text: 'Basic Configuration Structure', link: '/custom/basics' },
                { text: 'Model Structure', link: '/custom/model' },
                { text: 'Training and Validation Pipelines', link: '/custom/pipelines' },
                { text: 'Optimizers', link: '/custom/optimizer' },
            ]
        },
        {
            text: 'Datasets',
            collapsed: false,
            items: [
                { text: 'Public Datasets', link: '/datasets/public' },
                { text: 'Custom Datasets', link: '/datasets/custom' },
                { text: 'Dataset Formats and Extensions', link: '/datasets/extension' },
            ]
        },
        {
            text: 'Community',
            collapsed: false,
            items: [
                { text: 'FAQs', link: '/community/faqs' },
                { text: 'Reference Documentation', link: '/community/reference' },
                { text: 'Contribution Guide', link: '/community/contributing' },
                { text: 'Open Source License', link: '/community/license' }
            ]
        }
    ]
}
