/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      logger.h
* @brief     logger相关定义
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-04-19
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#ifndef LOGGER__H
#define LOGGER__H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include "hx_drv_timer.h"

#define ERROR_FILE_MAGIC1 0xfb321e68
#define ERROR_FILE_MAGIC2 0x1cf2ac4d
#define ERROR_DESCRIPTION_MAX 36
#define ERROR_FILE_MAX 64

    typedef enum
    {
        ERROR_NONE = 0,
        ERROR_MODEL_INVALID,
        ERROR_MODEL_PARSE,
        ERROR_MEMORY_ALLOCATION,
        ERROR_DATA_PRE,
        ERROR_DATA_POST,
        ERROR_ALGO_INIT,
        ERROR_ALGO_MISMATCH,
        ERROR_ALGO_PARM_INVALID,
        ERROR_ALGO_INVOKE,
        ERROR_ALGO_GET_RESULT,
        ERROR_SENSOR_NO_SUPPORT,
        ERROR_SENSOR_PARM_INVALID,
        ERROR_CAM_INIT,
        ERROR_CAM_DEINIT,
        ERROR_CAM_PARM_INVALID,
        ERROR_IMU_INIT,
        ERROR_WDT_TIMEOUT,
        ERROR_CMD_CRC,
        ERROR_MAX
    } ERROR_T;

    typedef struct
    {
        uint32_t uptime;
        uint8_t code;
        char description[ERROR_DESCRIPTION_MAX];
    } error_record_t;

    typedef struct
    {
        uint32_t start_magic1;
        uint32_t start_magic2;
        uint8_t length;
        error_record_t record[ERROR_FILE_MAX];
        uint32_t end_magic1;
        uint32_t end_magic2;
    } error_file_t;

    volatile extern uint8_t g_error;

    void error_file_show();
    int8_t error_file_load();
    int8_t error_file_stroge();
    int8_t error_file_clear();
    void get_error(char *log, ERROR_T error_code, uint16_t max_length);
    void logger(const char *fmt, ...);
    void logger_error(uint8_t error, const char *fmt, ...);

#if LOGGER_LEVEL > 3
#define _ENTRY                               \
    uint32_t _fun_tick = board_get_cur_us(); \
    logger("enty: %s\n", __FUNCTION__)
#define _EXIT \
    logger("exit: %s take: %d\n", __FUNCTION__, board_get_cur_us() - _fun_tick)
#else
#define _ENTRY
#define _EXIT
#endif

#if LOGGER_LEVEL > 2
#define LOGGER_INFO(fmt, ...) \
    logger(fmt, ##__VA_ARGS__)
#else
#define LOGGER_INFO(fmt, ...)
#endif
#if LOGGER_LEVEL > 1
#define LOGGER_WARNING(fmt, ...) \
    logger("[WARNING]");         \
    logger(fmt, ##__VA_ARGS__)
#else
#define LOGGER_WARNING(fmt, ...)
#endif
#if LOGGER_LEVEL > 0
#define LOGGER_ERROR(error, fmt, ...) \
    logger_error(error, fmt, ##__VA_ARGS__)
#else
#define LOGGER_ERROR(fmt, ...)
#endif

#ifdef __cplusplus
}
#endif

#endif
