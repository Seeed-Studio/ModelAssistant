import { defineConfig } from 'vitepress'

export default defineConfig({
    lang: 'en-US',
    description: 'Seeed Studio EdgeLab is an open-source project focused on embedded AI.',

    themeConfig: {
        nav: nav(),
        sidebar: { '/': sidebar() },

        editLink: {
            pattern: 'https://github.com/Seeed-Studio/Edgelab/edit/main/docs/:path',
            text: 'Suggest changes to this page'
        },

        footer: {
            message: 'Released under the MIT License',
            copyright: 'Copyright © 2023-Present Seeed Studio & EdgeLab Contributors',
        }
    }
})

function nav() {
    return [
        { text: 'Home', link: '/' },
        { text: 'Documentation', link: '/introduction/what_is_edgelab' }
    ]
}

function sidebar() {
    return [
        {
            text: 'Introduction',
            collapsed: false,
            items: [
                { text: 'What is EdgeLab?', link: '/introduction/what_is_edgelab' },
                { text: 'Quick Start', link: '/introduction/quick_start' },
                { text: 'Installation', link: '/introduction/installation' }
            ]
        },
        {
            text: 'Tutorials',
            collapsed: false,
            items: [
                { text: 'Config', link: '/tutorials/config' },
                { text: 'Datasets', link: '/tutorials/datasets' },
                {
                    text: 'Training',
                    link: '/tutorials/training/overview',
                    items: [
                        { text: 'FOMO Model', link: '/tutorials/training/fomo' },
                        { text: 'PFLD Model', link: '/tutorials/training/pfld' },
                        { text: 'YOLOv5 Model', link: '/tutorials/training/yolov5' },

                    ]
                },
                {
                    text: 'Export',
                    link: '/tutorials/export/overview',
                    items: [
                        { text: 'PyTorch to ONNX', link: '/tutorials/export/pytorch_2_onnx' },
                        { text: 'PyTorch to TFLite', link: '/tutorials/export/pytorch_2_tflite' }
                    ]
                }
            ]
        },
        {
            text: 'Examples',
            collapsed: false,
            link: '/examples/examples',
            items: [
                {
                    text: 'ESP32 - Deploy',
                    link: '/examples/esp32/deploy',
                    items: [
                        { text: 'ESP32 Mask Detection', link: '/examples/esp32/mask_detection' },
                        { text: 'ESP32 Meter Reader', link: '/examples/esp32/meter_reader' }
                    ]
                },
                {
                    text: 'Grove - Deploy',
                    link: '/examples/grove/deploy',
                    items: [
                        { text: 'Grove Mask Detection', link: '/examples/grove/mask_detection' },
                        { text: 'Grove Meter Reader', link: '/examples/grove/meter_reader' },
                        { text: 'Grove Digital Meter', link: '/examples/grove/digital_meter' }
                    ]
                }
            ]
        },
        {
            text: 'Edge Impulse',
            collapsed: false,
            items: [
                {
                    text: 'Edge Impulse Learning Blocks',
                    link: '/edgeimpulse/ei_ml_blocks',
                },
            ]
        },
        {
            text: 'Community',
            collapsed: false,
            items: [
                { text: 'FAQs', link: '/community/faqs' },
                { text: 'Reference', link: '/community/reference' },
                { text: 'Contribution Guidelines', link: '/community/contributing' },
                { text: 'Copyrights and Licenses', link: '/community/licenses' }
            ]
        }
    ]
}
