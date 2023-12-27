# 部署示例

ModelAssistant是一个提供No-Code可视化模型部署工具和基于CPP的SDK的开源项目。它使用户能够轻松地将模型部署到不同的平台上，无需编写复杂的代码。

目前支持的平台包括：
| 设备 | SenseCraft-Web-Toolkit | ModelAssistant-Micro SDK |
| :--- | :--- | :--- |
| [Xiao ESP32S3](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html) | ✅ | ✅ |
| [Grove Vision AI](https://www.seeedstudio.com/Grove-Vision-AI-Module-p-5457.html) | 🔜 | 🔜 |

## SenseCraft-Web-Toolkit

SenseCraft-Web-Toolkit是ModelAssistant提供的可视化模型部署工具。使用该工具，用户可以通过简单的操作将模型部署到各种平台上。这个工具提供了用户友好的界面，不需要编写任何代码。

[SenseCraft-Web-Toolkit](https://seeed-studio.github.io/SenseCraft-Web-Toolkit/)的主要特点包括：

- 可视化操作界面，无需编码技能
- 快速部署模型到不同的平台
- 支持TFLite格式的模型

Step 1. 打开SenseCraft-Web-Toolkit网站

<!-- <div align="center">
  <a href="https://seeed-studio.github.io/SenseCraft-Web-Toolk"><img width="10%" src="/public/images/ModelAssistant-Hero.png"/></a>
</div> -->

Step 2. 连接设备到电脑

使用带有数据传输功能的数据线将您的设备连接到您的电脑。

Step 3. 选择并连接你的设备

再进入SenseCraft-Web-Toolkit的主页后，我们需要首先连接设备，请点击连接按钮。

![step3-1](/static/deploy/step3-1.png)

然后，浏览器将弹出一个窗口。我们需要在此处选择正确的XIAO端口。对于Windows系统，该端口通常以COM开头，而对于MacOS系统，该端口通常以/dev/tty开头，并且会带有USB JTAG字样。如果您不确定正确的端口是什么，请在连接设备后刷新此页面，然后再次点击连接按钮，您将看到新的端口出现在下拉列表中。

![step3-2](/static/deploy/step3-2.png)

Step 4. 选择你的模型

一旦连接按钮变为红色的断开连接按钮，我们可以从“可供使用的AI模型”列表中选择模型。在这里，我选择了人脸识别作为演示。选择后，点击发送按钮并等待几秒钟。

![step4-1](/static/deploy/step4-1.png)

Step 5. 部署你的模型

![step5-1](/static/deploy/step5-1.png)

最后，我们来到预览部分，在右上角单击一次停止按钮，然后点击调用按钮，如果一切顺利，您可以看到实时屏幕效果。

![step5-2](/static/deploy/step5-2.png)

## ModelAssistant-Micro SDK

ModelAssistant还提供了基于CPP的SDK，名为ModelAssistant-Micro，使用户能够将模型部署到自己的项目中。通过集成ModelAssistant-Micro，用户可以方便地在自己的应用程序中使用部署好的模型。

[ModelAssistant-Micro](https://github.com/Seeed-Studio/ModelAssistant-Micro) SDK的特点包括：

- 基于CPP，适用于各种嵌入式系统和平台
- 提供简单而强大的API，方便用户进行模型调用和推理
- 支持TFLite格式的模型

## [Grove AI](./grove/deploy.md)

- [mask_detection](./grove/mask_detection.md)： 面罩检测
- [meter_reading](./grove/meter_reader.md)： 指针式仪表读数
- [digital_meter](./grove/digital_meter.md)： 数字式仪表读数

## [ESP32](./esp32/deploy.md)

- [mask_detection](./esp32/mask_detection.md)： 面罩检测
- [meter_reading](./esp32/meter_reader.md)： 指针式仪表读数

::: tip
更多的例子即将到来，敬请期待。
:::
