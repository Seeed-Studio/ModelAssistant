#ifndef SENSOR_CORE_H
#define SENSOR_CORE_H

#include <stdio.h>
#include "grove_ai_config.h"
#include "logger.h"

typedef enum {
    SENSOR_CAMERA = 0,
    SENSOR_MIC,
    SENSOR_IMU,
    SENSOR_MAX,
}Sensor_Type_List;

typedef struct {
    uint16_t width;
    uint16_t height;
}Camera_Cfg;

typedef struct {

}Mic_Cfg;

typedef struct {

}Imu_Cfg;

typedef struct {
    uint8_t sensor_type;
    union{
        Camera_Cfg camera_cfg;
        Mic_Cfg mic_cfg;
        Imu_Cfg imu_cfg;
    }data;
}Sensor_Cfg_t;

ERROR_T sensor_init(Sensor_Cfg_t *sensor_cfg_t);
ERROR_T sensor_deinit(void);

#endif
