/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      internal_flash.c
* @brief     外部flash驱动
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-04-19
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#include <stdlib.h>

#include "internal_flash.h"

#include "spi_master_protocol.h"
#include "hx_drv_spi_m.h"

int8_t internal_flash_read(uint32_t addr, void *data, uint32_t length)
{
    if (data == NULL)
    {
        return -1;
    }

    if (hx_drv_spi_flash_open(0) != 0)
    {
        return -1;
    }

    if (hx_drv_spi_flash_open_speed(50000000, 400000000) != 0)
    {
        return -1;
    }

    if (hx_drv_spi_flash_protocol_read(0, addr, (uint32_t)data, length, 4) != 0)
    {
        return -1;
    }

    if (hx_drv_spi_flash_close(0) != 0)
    {
        return -1;
    }

    return 0;
}

int8_t internal_flash_write(uint32_t addr, void *data, uint32_t length)
{
    if (data == NULL)
    {
        return -1;
    }

    if (hx_drv_spi_flash_open(0) != 0)
    {
        return -1;
    }

    if (hx_drv_spi_flash_open_speed(50000000, 400000000) != 0)
    {
        return -1;
    }

    uint32_t sector_size = length / INTERNAL_FLASH_SECTOR_SIZE + length % INTERNAL_FLASH_SECTOR_SIZE ? 1 : 0;
    for (int i = 0; i < sector_size; i++)
    {
        uint32_t sector_addr = addr + i * INTERNAL_FLASH_SECTOR_SIZE;
        if (hx_drv_spi_flash_protocol_erase_sector(0, sector_addr) != 0)
        {
            return -1;
        }
    }

    if (hx_drv_spi_flash_protocol_write(0, addr, (uint32_t)data, length, 4) != 0)
    {
        return -1;
    }

    if (hx_drv_spi_flash_close(0) != 0)
    {
        return -1;
    }

    return 0;
}

int8_t internal_flash_clear(uint32_t addr, uint32_t length)
{

    if (hx_drv_spi_flash_open(0) != 0)
    {
        return -1;
    }

    if (hx_drv_spi_flash_open_speed(50000000, 400000000) != 0)
    {
        return -1;
    }

    uint32_t sector_size = length / INTERNAL_FLASH_SECTOR_SIZE + length % INTERNAL_FLASH_SECTOR_SIZE ? 1 : 0;
    for (int i = 0; i < sector_size; i++)
    {
        uint32_t sector_addr = addr + i * INTERNAL_FLASH_SECTOR_SIZE;
        if (hx_drv_spi_flash_protocol_erase_sector(0, sector_addr) != 0)
        {
            return -1;
        }
    }

    if (hx_drv_spi_flash_close(0) != 0)
    {
        return -1;
    }

    return 0;
}