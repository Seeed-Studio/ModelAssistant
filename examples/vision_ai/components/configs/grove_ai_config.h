/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      grove_ai_config.h
* @brief     grove ai project configuration file
* @author    jian xiong (953308023@qq.com)
* @date      2022-04-24
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/
#ifndef GROVE_AI_CONFIG_H
#define GROVE_AI_CONFIG_H

#define GROVE_AI_FAMILY_MAIN_VER 0x01
#define GROVE_AI_FAMILY_SUB_VER 0x30

#define TENSOR_ARENA_SIZE 600 * 1024

#define SENSOR_WIDTH_DEFAULT 192
#define SENSOR_HEIGHT_DEFAULT 192

#define ALGO_CONFIG_ADDR 0x1FE000
#define ERROR_FILE_ADDR 0x1FF000

#define USE_WEBUSB
#define USE_I2C_SLAVE

#define I2C_SLAVE_ADDR 0x62

#define LOGGER_LEVEL 3


#ifdef SENSECAP_VISION_AI
#include "./boards/sensecap_vision_ai_config.h"
#elif defined GROVE_VISION_AI
#include "./boards/grove_vision_ai_config.h"
#endif


#define IGNORE_FIRST_X_PICS 0

#endif
