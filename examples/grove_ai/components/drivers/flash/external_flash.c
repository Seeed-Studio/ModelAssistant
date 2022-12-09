/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      external_flash.c
* @brief     外部flash驱动
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-04-19
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#include "hx_drv_iomux.h"
#include "hx_drv_spi_m.h"

#include "external_flash.h"

#ifdef FLASH_ENABLE_PIN
static void external_flash_power_on()
{
    hx_drv_iomux_set_pmux(FLASH_ENABLE_PIN, 3);
    hx_drv_iomux_set_outvalue(FLASH_ENABLE_PIN, FLASH_ENABLE_STATE);
    return;
}
static void external_flash_power_off()
{
    hx_drv_iomux_set_pmux(FLASH_ENABLE_PIN, 3);
    hx_drv_iomux_set_outvalue(FLASH_ENABLE_PIN, 1 - FLASH_ENABLE_STATE);
    return;
}
#else
static void external_flash_power_on()
{
    return;
}
static void external_flash_power_off()
{

    return;
}
#endif

int8_t external_flash_xip_enable()
{
    uint8_t flash_info[3] = "0";
    external_flash_power_on();
    board_delay_ms(100);
    DEV_SPI_PTR dev_spi_m;
    dev_spi_m = hx_drv_spi_mst_get_dev(USE_DW_SPI_MST_1);
    dev_spi_m->spi_open(DEV_MASTER_MODE, 50000000, 400000000); // master mode, spiclock, cpuclock
    dev_spi_m->flash_id(flash_info);
    dev_spi_m->flash_set_xip(true, SPI_M_MODE_QUAD);

    return 0;
}

int8_t external_flash_xip_disable()
{

    DEV_SPI_PTR dev_spi_m;
    dev_spi_m = hx_drv_spi_mst_get_dev(USE_DW_SPI_MST_1);
    dev_spi_m->flash_set_xip(false, SPI_M_MODE_QUAD);
    dev_spi_m->spi_close(); // master mode, spiclock, cpuclock
    external_flash_power_off();
    return 0;
}
