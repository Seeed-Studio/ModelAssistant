/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      debugger.h
* @brief     debugger
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-05-17
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#ifndef DEBUGGER__H
#define DEBUGGER__H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif
    void debugger_init();
    bool debugger_available();
    void debugger_task();
#ifdef __cplusplus
}
#endif
#endif