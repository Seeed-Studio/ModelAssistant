/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      sensecap_vision_ai_config.h
* @brief     sensecap vision_ai config
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-05-17
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/
#ifndef SENSECAP_VISION_AI_CONFIG_H
#define SENSECAP_VISION_AI_CONFIG_H

#define GROVE_AI_FAMILY_MAIN_ID 0x02
#define GROVE_AI_FAMILY_SUB_ID 0x00

#define FLASH_ENABLE_PIN IOMUX_PGPIO0
#define FLASH_ENABLE_STATE 1

#define CAMERA_ENABLE_PIN IOMUX_PGPIO9
#define CAMERA_ENABLE_STATE 1

#define I2C_SYNC_PIN IOMUX_PGPIO4
#define I2C_SYNC_STATE 1

#define WEBUSB_SYNC_PIN IOMUX_PGPIO8
#define WEBUSB_SYNC_STATE 1

#define DEBUGGER_ATTACH_PIN IOMUX_PGPIO1
#define DEBUGGER_ATTACH_STATE 1

#define USE_CAMERA
#define HM0360_CAMERA

//#define EXTERNAL_LDO // 开启外部LDO模块 需要配合硬件实现

#define VISION_ROTATION ROTATION_LEFT

#define CMD_READ_CRC
#define CMD_WRITE_CRC

#endif
