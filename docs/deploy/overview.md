# Model Deployment

SSCMA is an open-source project that provides a No-Code visual model deployment tool and a CPP-based SDK. It allows users to easily deploy models to different platforms without writing complex code.

The currently supported platforms include:

| Device | SenseCraft-Web-Toolkit | SSCMA-Micro SDK |
| :--- | :--- | :--- |
| [Xiao ESP32S3](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html) | ✅ | ✅ |
| [Grove Vision AI](https://www.seeedstudio.com/Grove-Vision-AI-Module-p-5457.html) | 🔜 | 🔜 |

## SenseCraft-Web-Toolkit

SenseCraft-Web-Toolkit is a visual model deployment tool provided by SSCMA. With this tool, users can easily deploy models to various platforms through simple operations. The tool provides a user-friendly interface and does not require any coding.

The main features of [SenseCraft-Web-Toolkit](https://seeed-studio.github.io/SenseCraft-Web-Toolkit/) include:

- Visual interface without coding skills
- Quick deployment of models to different platforms
- Support for TFLite models

Step 1. Open the SenseCraft-Web-Toolkit website

<div align="center">
  <a href="https://seeed-studio.github.io/SenseCraft-Web-Toolk"><img width="10%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Hero.png"/></a>
</div>

Step 2. Connect the device to your computer

Connect your device to your computer using a data cable with data transfer capability.

Step 3. Select and connect your device

After entering the homepage of SenseCraft-Web-Toolkit, we need to connect the device first. Please click the connect button.

![step3-1](https://files.seeedstudio.com/sscma/docs/static/deploy/step3-1.png)

Then, a window will pop up in the browser. We need to select the correct Xiao port here. For Windows systems, the port usually starts with COM, while for MacOS systems, the port usually starts with /dev/tty and has the USB JTAG label. If you are unsure about the correct port, refresh this page after connecting the device and click the connect button again, and you will see the new port appearing in the dropdown list.

![step3-2](https://files.seeedstudio.com/sscma/docs/static/deploy/step3-2.png)

Step 4. Choose your model

Once the connect button turns into a red disconnect button, we can select the model from the "Available AI Models" list. Here, I chose face recognition as a demonstration. After selection, click the send button and wait for a few seconds.

![step4-1](https://files.seeedstudio.com/sscma/docs/static/deploy/step4-1.png)

Step 5. Deploy your model

![step5-1](https://files.seeedstudio.com/sscma/docs/static/deploy/step5-1.png)

Finally, we come to the preview section. Click the stop button once in the upper right corner, and then click the invoke button. If everything goes well, you can see real-time screen effects.

![step5-2](https://files.seeedstudio.com/sscma/docs/static/deploy/step5-2.png)

## SSCMA-Micro SDK

SSCMA also provides a CPP-based SDK called SSCMA-Micro, enabling users to deploy models into their own projects. By integrating SSCMA-Micro, users can conveniently use the deployed models in their applications.

The features of the [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro) SDK include:

- CPP-based, suitable for various embedded systems and platforms
- Provides a simple yet powerful API for model invocation and inference
- Support for TFLite models

### [Grove AI](./grove/deploy)

- [mask_detection](./grove/mask_detection): Face mask detection
- [meter_reading](./grove/meter_reader): Pointer meter reading
- [digital_meter](./grove/digital_meter): Digital meter reading

### [ESP32](./esp32/deploy)

- [mask_detection](./esp32/mask_detection): Face mask detection
- [meter_reading](./esp32/meter_reader): Pointer meter reading

:::tip
More examples are coming soon, stay tuned.
:::
