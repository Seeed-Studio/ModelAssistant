/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      meter.c
* @brief
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-12-09
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>
#include "hx_drv_timer.h"
#include "grove_ai_config.h"
#include "hx_drv_webusb.h"
#include "logger.h"
#include "debugger.h"
#include "datapath.h"
#include "sensor_core.h"
#include "external_flash.h"

char preview[1024];

ERROR_T hardware_init()
{

    ERROR_T ret = ERROR_NONE;
    Sensor_Cfg_t sensor_cfg_t = {
        .sensor_type = SENSOR_CAMERA,
        .data.camera_cfg.width = 240,
        .data.camera_cfg.height = 240,
    };

    ret = datapath_init(sensor_cfg_t.data.camera_cfg.width,
                        sensor_cfg_t.data.camera_cfg.height);
    if (ret != ERROR_NONE)
    {
        return ret;
    }
    ret = sensor_init(&sensor_cfg_t);
    if (ret != ERROR_NONE)
    {
        return ret;
    }

    return ERROR_NONE;
}

void main(void)
{
    int ercode = 0;

    hx_drv_timer_init();
    debugger_init();

    external_flash_xip_enable();

    communication_init();

    ercode = tflitemicro_algo_init();
    if (ercode != 0)
        LOGGER_INFO("tflitemicro_algo_init() error\n");

    hardware_init();

    uint32_t frame = 0;

    for (;;)
    {
        LOGGER_INFO("Frame: %d\r", frame++);
        datapath_start_work();
        // temp
        while (!datapath_get_img_state())
        {
        }
        if (ercode == 0)
        {
            uint32_t raw_img_addr = datapath_get_yuv_img_addr();
            tflitemicro_algo_run(raw_img_addr, 240, 240);
        }
        volatile uint32_t jpeg_addr;
        volatile uint32_t jpeg_size;
        datapath_get_jpeg_img(&jpeg_addr, &jpeg_size);
        hx_drv_webusb_write_vision(jpeg_addr, jpeg_size);
        if (ercode == 0)
        {
            tflitemicro_algo_get_preview(preview, 1024);
            if (strlen(preview) > 0)
            {
                LOGGER_INFO("%s\r", preview);
                hx_drv_webusb_write_text(preview, strlen(preview));
            }
        }
    }

    return;
}
