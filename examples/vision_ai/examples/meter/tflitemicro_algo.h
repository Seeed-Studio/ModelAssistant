/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      algo_meter.h
* @brief
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-08-22
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#ifndef ALGO_METER__H
#define ALGO_METER__H


#ifdef __cplusplus
extern "C"
{
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "embARC_debug.h"
#include "hx_drv_pmu.h"
#include "powermode.h"
#endif

    typedef struct
    {
        uint8_t x;
        uint8_t y;
    } meter_t;

    int tflitemicro_algo_init();
    int tflitemicro_algo_run(uint32_t img, uint32_t w, uint32_t h);
    int tflitemicro_algo_get_preview(char *preview, uint16_t max_length);
    void tflitemicro_algo_exit();

#ifdef __cplusplus
}
#endif

#endif
