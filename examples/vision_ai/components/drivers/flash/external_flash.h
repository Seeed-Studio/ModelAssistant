/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      external_flash.h
* @brief     外部flash驱动
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-05-10
* @version   v1.1
* @changelog  2022-05-10: 修改 flash_xip 为 external_flash
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/
#ifndef EXTERNL_FLASH__H
#define EXTERNL_FLASH__H

#include <stdint.h>
#include <stdbool.h>

#include "grove_ai_config.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int8_t external_flash_xip_enable();
    int8_t external_flash_xip_disable();

#ifdef __cplusplus
}
#endif

#endif
