/**
*****************************************************************************************
*     Copyright(c) 2022, Seeed Technology Corporation. All rights reserved.
*****************************************************************************************
* @file      logger.c
* @brief     logger相关定义
* @author    Hongtai Liu (lht856@foxmail.com)
* @date      2022-04-19
* @version   v1.0
**************************************************************************************
* @attention
* <h2><center>&copy; COPYRIGHT 2022 Seeed Technology Corporation</center></h2>
**************************************************************************************
*/

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "grove_ai_config.h"
#include "logger.h"
#include "debugger.h"
#include "embARC_debug.h"
#include "hx_drv_timer.h"
#include "hx_drv_iomux.h"
#include "internal_flash.h"

volatile static error_file_t erro_file;
volatile uint8_t g_error = ERROR_NONE;

#define LOG_FORMATE "{\"type\":\"log\", \"algorithm\":%d, \"model\":%d, \"code\":%d, \"description\":\"%s\"}"

static const char const *error_descriptions[] = {
    "NONE",
    "Model Invalid Or Not Existent.",
    "Model Parsing Failure.",
    "Memory Allocation Failure.",
    "Pre-processed Data Failure.",
    "Post-processed Data Failure.",
    "Algorithm Initialization Failure.",
    "Algorithm Mismatch With the Model.",
    "Algorithm Parameter Invalid.",
    "Algorithm Invoke Failure.",
    "Algorithm Get Results Failure.",
    "Sensor Not Supported Yet.",
    "Sensor Parameter Invalid.",
    "Camera Initialization Failure.",
    "Camera De-Initialization Failure.",
    "Camera Parameter Invalid.",
    "IMU Initialization Failure.",
    "Watchdog Timeout.",
    "Command CRC checksum error.",
    "Unknown Error.",
};

void error_file_show_raw()
{
    uint8_t *temp = (uint8_t *)&erro_file;
    for (int i = 0; i < sizeof(erro_file); i++)
    {
        EMBARC_PRINTF("%c", *temp++);
    }
    return;
}

void error_file_show()
{
    EMBARC_PRINTF("\n************error record start****************\n");
    for (int i = 0; i < erro_file.length; i++)
    {
        EMBARC_PRINTF("[uptime: %d, error code: %d]: %s\n", erro_file.record[i].uptime, erro_file.record[i].code, erro_file.record[i].description);
    }
    EMBARC_PRINTF("\n************error record end****************\n");
    return;
}

void error_file_write(uint8_t error)
{
    if (erro_file.length == ERROR_FILE_MAX)
    {
        erro_file.length = ERROR_FILE_MAX - 1;
        for (int i = 0; i < erro_file.length; i++)
        {
            erro_file.record[i].code = erro_file.record[i + 1].code;
            erro_file.record[i].uptime = erro_file.record[i + 1].uptime;
            memcpy(erro_file.record[i].description, erro_file.record[i + 1].description, ERROR_DESCRIPTION_MAX);
        }
    }

    erro_file.record[erro_file.length].code = error;
    erro_file.record[erro_file.length].uptime = board_get_cur_us() / 1000;
    memcpy(erro_file.record[erro_file.length].description, error_descriptions[error], ERROR_DESCRIPTION_MAX);
    erro_file.length++;

    error_file_stroge();
}
int8_t error_file_clear()
{
    memset((void *)&erro_file, 0, sizeof(erro_file));
    erro_file.length = 0;
    return error_file_stroge();
}

int8_t error_file_load()
{
    memset((void *)&erro_file, 0, sizeof(erro_file));

    if (internal_flash_read(ERROR_FILE_ADDR, &erro_file, sizeof(erro_file)) != 0)
    {
        return -1;
    }

    if (erro_file.start_magic1 != ERROR_FILE_MAGIC1 || erro_file.start_magic2 != ERROR_FILE_MAGIC2)
    {
        memset((void *)&erro_file, 0, sizeof(erro_file));
        erro_file.length = 0;
        error_file_stroge();
        return -1;
    }

    if (erro_file.end_magic1 != ERROR_FILE_MAGIC1 || erro_file.end_magic2 != ERROR_FILE_MAGIC2)
    {
        memset((void *)&erro_file, 0, sizeof(erro_file));
        erro_file.length = 0;
        error_file_stroge();
        return -1;
    }

    return 0;
}

int8_t error_file_stroge()
{
    erro_file.start_magic1 = ERROR_FILE_MAGIC1;
    erro_file.start_magic2 = ERROR_FILE_MAGIC2;
    erro_file.end_magic1 = ERROR_FILE_MAGIC1;
    erro_file.end_magic2 = ERROR_FILE_MAGIC2;

    if (internal_flash_write(ERROR_FILE_ADDR, &erro_file, sizeof(erro_file)) != 0)
    {
        return -1;
    }
    return 0;
}

void logger(const char *fmt, ...)
{
    if (!debugger_available())
    {
        return;
    }

    char print_buf[256] = {0};

    va_list args;
    va_start(args, fmt);
    int r = vsnprintf(print_buf, sizeof(print_buf), fmt, args);
    va_end(args);

    if (r > 0)
    {
        EMBARC_PRINTF("%s", print_buf);
    }
}

void logger_error(uint8_t error, const char *fmt, ...)
{
    char print_buf[256] = {0};

    va_list args;
    va_start(args, fmt);
    int r = vsnprintf(print_buf, sizeof(print_buf), fmt, args);
    va_end(args);

    if (r <= 0)
    {
        return;
    }

    if (error != ERROR_NONE)
    {
        g_error = error;
        error_file_write(error);
    }

    if (!debugger_available())
    {
        return;
    }

    EMBARC_PRINTF("[ERROR: %s]%s", error_descriptions[error], print_buf);
}

void get_error(char *log, ERROR_T error_code, uint16_t max_length)
{
    if (log == NULL)
    {
        return;
    }
    if (error_code == ERROR_MAX)

    {
        error_code = ERROR_MAX;
    }
    memset(log, 0, max_length);

    snprintf(log, max_length, LOG_FORMATE, tflitemicro_algo_algo_index(), tflitemicro_algo_model_index(), error_code, error_descriptions[error_code]);
}
