#include <stdio.h>
#include <stdlib.h>

#include "grove_ai_config.h"
#include "logger.h"

#include "hx_drv_lsm6ds3.h"

#include "lsm6ds3.h"

static DEV_LSM6DS3_PTR lsm6ds3 = NULL;

static ERROR_T lsm6ds3_init(void)
{
    lsm6ds3 = hx_drv_lsm6ds3_init(SS_IIC_0_ID);
    if (lsm6ds3 == NULL)
    {
        return ERROR_IMU_INIT;
    }
    if (!hx_drv_lsm6ds3_begin(lsm6ds3))
    {
        free(lsm6ds3);
        lsm6ds3 = NULL;
        return ERROR_IMU_INIT;
    }
    LOGGER_INFO("imu init sucess!");
    return ERROR_NONE;
}

static ERROR_T lsm6ds3_deinit(void)
{

    if (lsm6ds3 != NULL)
    {
        free(lsm6ds3);
        lsm6ds3 = NULL;
    }
    return ERROR_NONE;
}

static bool lsm6ds3_acc_available(void)
{
    return hx_drv_lsm6ds3_acc_available(lsm6ds3);
}

static bool lsm6ds3_gyro_available(void)
{
    return hx_drv_lsm6ds3_gyro_available(lsm6ds3);
}

static float lsm6ds3_get_acc_x(void)
{
    return hx_drv_lsm6ds3_read_acc_x(lsm6ds3);
}

static float lsm6ds3_get_acc_y(void)
{
    return hx_drv_lsm6ds3_read_acc_y(lsm6ds3);
}

static float lsm6ds3_get_acc_z(void)
{
    return hx_drv_lsm6ds3_read_acc_z(lsm6ds3);
}

static float lsm6ds3_get_gyro_x(void)
{
    return hx_drv_lsm6ds3_read_gyro_x(lsm6ds3);
}

static float lsm6ds3_get_gyro_y(void)
{
    return hx_drv_lsm6ds3_read_gyro_y(lsm6ds3);
}

static float lsm6ds3_get_gyro_z(void)
{
    return hx_drv_lsm6ds3_read_gyro_z(lsm6ds3);
}

const Imu_Hal_Struct lsm6ds3_driver = {
    .init = lsm6ds3_init,
    .deinit = lsm6ds3_deinit,
    .acc_available = lsm6ds3_acc_available,
    .gyro_available = lsm6ds3_gyro_available,
    .get_acc_x = lsm6ds3_get_acc_x,
    .get_acc_y = lsm6ds3_get_acc_y,
    .get_acc_z = lsm6ds3_get_acc_z,
    .get_gyro_x = lsm6ds3_get_gyro_x,
    .get_gyro_y = lsm6ds3_get_gyro_y,
    .get_gyro_z = lsm6ds3_get_gyro_z,
};
