#include "sensor_core.h"
#include "camera_core.h"
#include "hx_drv_pmu.h"
#include "logger.h"
#include "error_code.h"

ERROR_T sensor_init(Sensor_Cfg_t *sensor_cfg_t)
{
    ERROR_T ret = ERROR_NONE;

    if (sensor_cfg_t == NULL)
        return ERROR_SENSOR_PARM_INVALID;

    switch (sensor_cfg_t->sensor_type)
    {
#if defined(USE_CAMERA)
    case SENSOR_CAMERA:
        ret = camera_init(&(sensor_cfg_t->data.camera_cfg));
        if (ret == ERROR_NONE)
        {
            hx_drv_pmu_set_ctrl(PMU_SEN_INIT, 0);
        }
        break;
#endif
#if defined(USE_IMU)
    case SENSOR_IMU:
        ret = imu_init();
        break;
#endif
#if defined(USE_MIC)
    case SENSOR_MIC:
        ret = mic_init();
        break;
#endif
    default:
        ret = ERROR_NONE;
        break;
    }

    return ret;
}

ERROR_T sensor_deinit(void)
{
    int8_t ret = 0x00;
#if defined(USE_CAMERA)
    ret = camera_deinit();
    if (ret)
    {
        return ret;
    }
#endif

    return ERROR_NONE;
}
