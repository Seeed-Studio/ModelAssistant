/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      grove_vision_ai_config.h
* @brief     grove vision_ai config
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-05-17
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/
#ifndef GROVE_VISION_AI_CONFIG_H
#define GROVE_VISION_AI_CONFIG_H

#define GROVE_AI_FAMILY_MAIN_ID 0x01
#define GROVE_AI_FAMILY_SUB_ID 0x00

#define I2C_SYNC_PIN IOMUX_PGPIO0
#define I2C_SYNC_STATE 1

#define WEBUSB_SYNC_PIN IOMUX_PGPIO8
#define WEBUSB_SYNC_STATE 1

#define DEBUGGER_ATTACH_PIN IOMUX_PGPIO1
#define DEBUGGER_ATTACH_STATE 1

#define USE_CAMERA
#define OV2640_CAMERA

//#define USE_IMU
//#define IMU_LSM6DS3

#define CMD_IMU
#define CMD_GPIO

#define CMD_READ_CRC
#define CMD_WRITE_CRC


#define CAMERA_ENABLE_PIN IOMUX_PGPIO9
#define CAMERA_ENABLE_STATE 1

#define VISION_ROTATION ROTATION_UP

#endif
