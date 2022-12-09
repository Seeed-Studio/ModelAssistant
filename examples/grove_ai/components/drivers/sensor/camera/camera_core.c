#include "camera_core.h"
#include "hx_drv_dp.h"
#include "hx_drv_sensorctrl.h"
#include "sensor_dp_lib.h"
#include "hx_drv_inp.h"
#include "error_code.h"
#include "logger.h"
#include "sensor_core.h"

extern Camera_Hal_Struct ov2640_driver;
extern Camera_Hal_Struct hm0360_driver;

#if defined(OV2640_CAMERA)
Camera_Hal_Struct *camera_hal = &ov2640_driver;
#elif defined(HM0360_CAMERA)
Camera_Hal_Struct *camera_hal = &hm0360_driver;
#else
Camera_Hal_Struct *camera_hal = NULL;
#endif

ERROR_T camera_init(Camera_Cfg *camera_cfg)
{
    uint32_t sensor_id = 0x00;
    uint32_t result = DP_NO_ERROR;
    SENSORDPLIB_STREAM_E stream_type = SENSORDPLIB_STREAM_NONEAOS;

    // parameter check
    if (camera_cfg == NULL)
        return ERROR_CAM_PARM_INVALID;
    if (camera_hal == NULL)
    {

        return ERROR_CAM_PARM_INVALID;
    }
    if (camera_hal->power_init == NULL)
    {

        return ERROR_CAM_INIT;
    }
    if (camera_hal->get_sensor_id == NULL)
    {
        return ERROR_CAM_INIT;
    }
    if (camera_hal->sensor_cfg == NULL)
    {
        return ERROR_CAM_INIT;
    }
    if (camera_hal->set_output_size == NULL)
    {
        return ERROR_CAM_INIT;
    }

    // init the MCLK for camera
    result = hx_drv_dp_set_dp_clk_src(DP_CLK_SRC_XTAL_24M_POST);
    if (result)
    {
        return ERROR_CAM_INIT;
    }
    result = hx_drv_dp_set_mclk_src(DP_MCLK_SRC_INTERNAL, DP_MCLK_SRC_INT_SEL_XTAL);
    if (result)
    {
        return ERROR_CAM_INIT;
    }

    result = hx_drv_cis_init(camera_hal->xshutdown_pin, SENSORCTRL_MCLK_DIV1);
    if (result)
    {
        return ERROR_CAM_INIT;
    }

    // power init
    camera_hal->power_init();

    // get camera I2C ID
    if (0 != camera_hal->get_sensor_id(&sensor_id))
    {
        return ERROR_CAM_INIT;
    }

    LOGGER_INFO("Camera ID: 0x%x\n", sensor_id);

    // config the camera regs
    result = hx_drv_cis_setRegTable(camera_hal->sensor_cfg, camera_hal->sensor_cfg_len);
    if (result)
    {
        return ERROR_CAM_INIT;
    }

    // config the output size of camera
    result = camera_hal->set_output_size(camera_cfg->width, camera_cfg->height);
    if (result)
    {
        return ERROR_CAM_PARM_INVALID;
    }

    // INP and sensor control config
    if(sensor_id == 0x360)
        stream_type = SENSORDPLIB_STREAM_HM0360_CONT_MCLK;
    else
        stream_type = SENSORDPLIB_STREAM_NONEAOS;
    result = sensordplib_set_sensorctrl_inp(SENSORDPLIB_SENSOR_HM0360_MODE1,
                                            stream_type, camera_cfg->width, camera_cfg->height,
                                            INP_SUBSAMPLE_DISABLE);
    if (result)
    {
        return ERROR_CAM_INIT;
    }

    result = hx_drv_sensorctrl_set_MCLKCtrl(SENSORCTRL_MCLKCTRL_NONAOS);
    if (result)
    {
        return ERROR_CAM_INIT;
    }

    // config the xsleep pin
    if (camera_hal->xsleep_ctl == SENSORCTRL_XSLEEP_BY_SC)
    {
        result = hx_drv_sensorctrl_set_xSleepCtrl(SENSORCTRL_XSLEEP_BY_SC);
        if (result)
        {
            return ERROR_CAM_INIT;
        }
    }
    else
    {
        result = hx_drv_sensorctrl_set_xSleepCtrl(SENSORCTRL_XSLEEP_BY_CPU);
        if (result)
        {
            return ERROR_CAM_INIT;
        }
        result = hx_drv_sensorctrl_set_xSleep(1);
        if (result)
        {
            return ERROR_CAM_INIT;
        }
    }

    sensordplib_set_mclkctrl_xsleepctrl_bySCMode();

    LOGGER_INFO("camera init success!\n");

    return ERROR_NONE;
}

ERROR_T camera_deinit()
{
    // parameter check
    if (camera_hal == NULL)
    {
        return ERROR_CAM_DEINIT;
    }
    if (camera_hal->power_off == NULL)
    {
        return ERROR_CAM_DEINIT;
    }

    camera_hal->power_off();

    return ERROR_NONE;
}
