#include "grove_ai_config.h"
#include "logger.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "imu_core.h"

#if defined(IMU_LSM6DS3)
extern  const Imu_Hal_Struct lsm6ds3_driver;
const static Imu_Hal_Struct *imu_hal = &lsm6ds3_driver;
#else
static const Imu_Hal_Struct *imu_hal = NULL;
#endif

ERROR_T imu_init()
{
    if (imu_hal->init != NULL)
    {
        return imu_hal->init();
    }
    return ERROR_SENSOR_NO_SUPPORT;
}
ERROR_T imu_deinit()
{
    if (imu_hal->deinit != NULL)
    {
        return imu_hal->deinit();
    }
    return ERROR_SENSOR_NO_SUPPORT;
}

bool imu_acc_available()
{
    if (imu_hal->acc_available != NULL)
    {
        return imu_hal->acc_available();
    }
    else
    {
        return false;
    }
}

bool imu_gyro_available()
{
    if (imu_hal->gyro_available != NULL)
    {
        return imu_hal->gyro_available();
    }
    else
    {
        return false;
    }
}

float imu_get_acc_x()
{
    if (imu_hal->get_acc_x != NULL)
    {
        return imu_hal->get_acc_x();
    }
    else
    {
        return 0;
    }
}
float imu_get_acc_y()
{
    if (imu_hal->get_acc_y != NULL)
    {
        return imu_hal->get_acc_y();
    }
    else
    {
        return 0;
    }
}
float imu_get_acc_z()
{

    if (imu_hal->get_acc_z != NULL)
    {
        return imu_hal->get_acc_z();
    }
    else
    {
        return 0;
    }
}

float imu_get_gyro_x()
{
    if (imu_hal->get_gyro_x != NULL)
    {
        return imu_hal->get_gyro_x();
    }
    else
    {
        return 0;
    }
}
float imu_get_gyro_y()
{
    if (imu_hal->get_gyro_y != NULL)
    {
        return imu_hal->get_gyro_y();
    }
    else
    {
        return 0;
    }
}
float imu_get_gyro_z()
{
    if (imu_hal->get_gyro_z != NULL)
    {
        return imu_hal->get_gyro_z();
    }
    else
    {
        return 0;
    }
}
