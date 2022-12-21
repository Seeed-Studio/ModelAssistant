#ifndef IMU_CORE_H
#define IMU_CORE_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "grove_ai_config.h"
#include "logger.h"

typedef struct
{
    ERROR_T (*init)(void);
    ERROR_T (*deinit)(void);
    bool (*acc_available)(void);
    bool (*gyro_available)(void);
    float (*get_acc_x)(void);
    float (*get_acc_y)(void);
    float (*get_acc_z)(void);
    float (*get_gyro_x)(void);
    float (*get_gyro_y)(void);
    float (*get_gyro_z)(void);
} Imu_Hal_Struct;

ERROR_T imu_init(void);
ERROR_T imu_deinit(void);
bool imu_acc_available(void);
bool imu_gyro_available(void);
float imu_get_acc_x(void);
float imu_get_acc_y(void);
float imu_get_acc_z(void);
float imu_get_gyro_x(void);
float imu_get_gyro_y(void);
float imu_get_gyro_z(void);

#endif
