/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      internal_flash.h
* @brief     内部flash驱动
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-05-10
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#ifndef EXTERNL_FLASH__H
#define EXTERNL_FLASH__H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    #define INTERNAL_FLASH_SECTOR_SIZE 4096

    int8_t internal_flash_read(uint32_t addr, void *data, uint32_t length);
    int8_t internal_flash_write(uint32_t addr, void *data, uint32_t length);
    int8_t internal_flash_clear(uint32_t addr, uint32_t length);

#ifdef __cplusplus
}
#endif

#endif