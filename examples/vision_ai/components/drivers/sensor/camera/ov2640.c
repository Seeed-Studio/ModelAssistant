#include "ov2640.h"
#include "error_code.h"
#include "hx_drv_CIS_common.h"
#include "logger.h"

const HX_CIS_SensorSetting_t sensor_ov2640_setting[] = {
	{HX_CIS_I2C_Action_W, 0xff, 0x00},
	{HX_CIS_I2C_Action_W, 0x2c, 0xff},
	{HX_CIS_I2C_Action_W, 0x2e, 0xdf},
	{HX_CIS_I2C_Action_W, 0xff, 0x01},
	{HX_CIS_I2C_Action_W, 0x3c, 0x32},
	{HX_CIS_I2C_Action_W, 0x11, 0x00},
	{HX_CIS_I2C_Action_W, 0x09, 0x02},
	{HX_CIS_I2C_Action_W, 0x04, 0xd0},
	{HX_CIS_I2C_Action_W, 0x13, 0xe5},
	{HX_CIS_I2C_Action_W, 0x14, 0x48},
	{HX_CIS_I2C_Action_W, 0x2c, 0x0c},
	{HX_CIS_I2C_Action_W, 0x33, 0x78},
	{HX_CIS_I2C_Action_W, 0x3a, 0x33},
	{HX_CIS_I2C_Action_W, 0x3b, 0xfB},
	{HX_CIS_I2C_Action_W, 0x3e, 0x00},
	{HX_CIS_I2C_Action_W, 0x43, 0x11},
	{HX_CIS_I2C_Action_W, 0x16, 0x10},
	{HX_CIS_I2C_Action_W, 0x4a, 0x81},
	{HX_CIS_I2C_Action_W, 0x21, 0x99},
	{HX_CIS_I2C_Action_W, 0x24, 0x40},
	{HX_CIS_I2C_Action_W, 0x25, 0x38},
	{HX_CIS_I2C_Action_W, 0x26, 0x82},
	{HX_CIS_I2C_Action_W, 0x5c, 0x00},
	{HX_CIS_I2C_Action_W, 0x63, 0x00},
	{HX_CIS_I2C_Action_W, 0x46, 0x3f},
	{HX_CIS_I2C_Action_W, 0x0c, 0x3c},
	{HX_CIS_I2C_Action_W, 0x61, 0x70},
	{HX_CIS_I2C_Action_W, 0x62, 0x80},
	{HX_CIS_I2C_Action_W, 0x7c, 0x05},
	{HX_CIS_I2C_Action_W, 0x20, 0x80},
	{HX_CIS_I2C_Action_W, 0x28, 0x30},
	{HX_CIS_I2C_Action_W, 0x6c, 0x00},
	{HX_CIS_I2C_Action_W, 0x6d, 0x80},
	{HX_CIS_I2C_Action_W, 0x6e, 0x00},
	{HX_CIS_I2C_Action_W, 0x70, 0x02},
	{HX_CIS_I2C_Action_W, 0x71, 0x94},
	{HX_CIS_I2C_Action_W, 0x73, 0xc1},
	{HX_CIS_I2C_Action_W, 0x3d, 0x34},
	{HX_CIS_I2C_Action_W, 0x5a, 0x57},
	{HX_CIS_I2C_Action_W, 0x12, 0x00},
	{HX_CIS_I2C_Action_W, 0x11, 0x00},
	{HX_CIS_I2C_Action_W, 0x17, 0x11},
	{HX_CIS_I2C_Action_W, 0x18, 0x75},
	{HX_CIS_I2C_Action_W, 0x19, 0x01},
	{HX_CIS_I2C_Action_W, 0x1a, 0x97},
	{HX_CIS_I2C_Action_W, 0x32, 0x36},
	{HX_CIS_I2C_Action_W, 0x03, 0x0f},
	{HX_CIS_I2C_Action_W, 0x37, 0x40},
	{HX_CIS_I2C_Action_W, 0x4f, 0xbb},
	{HX_CIS_I2C_Action_W, 0x50, 0x9c},
	{HX_CIS_I2C_Action_W, 0x5a, 0x57},
	{HX_CIS_I2C_Action_W, 0x6d, 0x80},
	{HX_CIS_I2C_Action_W, 0x6d, 0x38},
	{HX_CIS_I2C_Action_W, 0x39, 0x02},
	{HX_CIS_I2C_Action_W, 0x35, 0x88},
	{HX_CIS_I2C_Action_W, 0x22, 0x0a},
	{HX_CIS_I2C_Action_W, 0x37, 0x40},
	{HX_CIS_I2C_Action_W, 0x23, 0x00},
	{HX_CIS_I2C_Action_W, 0x34, 0xa0},
	{HX_CIS_I2C_Action_W, 0x36, 0x1a},
	{HX_CIS_I2C_Action_W, 0x06, 0x02},
	{HX_CIS_I2C_Action_W, 0x07, 0xc0},
	{HX_CIS_I2C_Action_W, 0x0d, 0xb7},
	{HX_CIS_I2C_Action_W, 0x0e, 0x01},
	{HX_CIS_I2C_Action_W, 0x4c, 0x00},
	{HX_CIS_I2C_Action_W, 0xff, 0x00},
	{HX_CIS_I2C_Action_W, 0xe5, 0x7f},
	{HX_CIS_I2C_Action_W, 0xf9, 0xc0},
	{HX_CIS_I2C_Action_W, 0x41, 0x24},
	{HX_CIS_I2C_Action_W, 0xe0, 0x14},
	{HX_CIS_I2C_Action_W, 0x76, 0xff},
	{HX_CIS_I2C_Action_W, 0x33, 0xa0},
	{HX_CIS_I2C_Action_W, 0x42, 0x20},
	{HX_CIS_I2C_Action_W, 0x43, 0x18},
	{HX_CIS_I2C_Action_W, 0x4c, 0x00},
	{HX_CIS_I2C_Action_W, 0x87, 0xd0},
	{HX_CIS_I2C_Action_W, 0x88, 0x3f},
	{HX_CIS_I2C_Action_W, 0xd7, 0x03},
	{HX_CIS_I2C_Action_W, 0xd9, 0x10},
	{HX_CIS_I2C_Action_W, 0xd3, 0x82},
	{HX_CIS_I2C_Action_W, 0xc8, 0x08},
	{HX_CIS_I2C_Action_W, 0xc9, 0x80},
	{HX_CIS_I2C_Action_W, 0x7d, 0x00},
	{HX_CIS_I2C_Action_W, 0x7c, 0x03},
	{HX_CIS_I2C_Action_W, 0x7d, 0x48},
	{HX_CIS_I2C_Action_W, 0x7c, 0x08},
	{HX_CIS_I2C_Action_W, 0x7d, 0x20},
	{HX_CIS_I2C_Action_W, 0x7d, 0x10},
	{HX_CIS_I2C_Action_W, 0x7d, 0x0e},
	{HX_CIS_I2C_Action_W, 0x90, 0x00},
	{HX_CIS_I2C_Action_W, 0x91, 0x0e},
	{HX_CIS_I2C_Action_W, 0x91, 0x1a},
	{HX_CIS_I2C_Action_W, 0x91, 0x31},
	{HX_CIS_I2C_Action_W, 0x91, 0x5a},
	{HX_CIS_I2C_Action_W, 0x91, 0x69},
	{HX_CIS_I2C_Action_W, 0x91, 0x75},
	{HX_CIS_I2C_Action_W, 0x91, 0x7e},
	{HX_CIS_I2C_Action_W, 0x91, 0x88},
	{HX_CIS_I2C_Action_W, 0x91, 0x8f},
	{HX_CIS_I2C_Action_W, 0x91, 0x96},
	{HX_CIS_I2C_Action_W, 0x91, 0xa3},
	{HX_CIS_I2C_Action_W, 0x91, 0xaf},
	{HX_CIS_I2C_Action_W, 0x91, 0xc4},
	{HX_CIS_I2C_Action_W, 0x91, 0xd7},
	{HX_CIS_I2C_Action_W, 0x91, 0xe8},
	{HX_CIS_I2C_Action_W, 0x91, 0x20},
	{HX_CIS_I2C_Action_W, 0x92, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x06},
	{HX_CIS_I2C_Action_W, 0x93, 0xe3},
	{HX_CIS_I2C_Action_W, 0x93, 0x02},
	{HX_CIS_I2C_Action_W, 0x93, 0x02},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x04},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x03},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x93, 0x00},
	{HX_CIS_I2C_Action_W, 0x96, 0x00},
	{HX_CIS_I2C_Action_W, 0x97, 0x08},
	{HX_CIS_I2C_Action_W, 0x97, 0x19},
	{HX_CIS_I2C_Action_W, 0x97, 0x02},
	{HX_CIS_I2C_Action_W, 0x97, 0x0c},
	{HX_CIS_I2C_Action_W, 0x97, 0x24},
	{HX_CIS_I2C_Action_W, 0x97, 0x30},
	{HX_CIS_I2C_Action_W, 0x97, 0x28},
	{HX_CIS_I2C_Action_W, 0x97, 0x26},
	{HX_CIS_I2C_Action_W, 0x97, 0x02},
	{HX_CIS_I2C_Action_W, 0x97, 0x98},
	{HX_CIS_I2C_Action_W, 0x97, 0x80},
	{HX_CIS_I2C_Action_W, 0x97, 0x00},
	{HX_CIS_I2C_Action_W, 0x97, 0x00},
	{HX_CIS_I2C_Action_W, 0xc3, 0xef},
	{HX_CIS_I2C_Action_W, 0xff, 0x00},
	{HX_CIS_I2C_Action_W, 0xba, 0xdc},
	{HX_CIS_I2C_Action_W, 0xbb, 0x08},
	{HX_CIS_I2C_Action_W, 0xb6, 0x24},
	{HX_CIS_I2C_Action_W, 0xb8, 0x33},
	{HX_CIS_I2C_Action_W, 0xb7, 0x20},
	{HX_CIS_I2C_Action_W, 0xb9, 0x30},
	{HX_CIS_I2C_Action_W, 0xb3, 0xb4},
	{HX_CIS_I2C_Action_W, 0xb4, 0xca},
	{HX_CIS_I2C_Action_W, 0xb5, 0x43},
	{HX_CIS_I2C_Action_W, 0xb0, 0x5c},
	{HX_CIS_I2C_Action_W, 0xb1, 0x4f},
	{HX_CIS_I2C_Action_W, 0xb2, 0x06},
	{HX_CIS_I2C_Action_W, 0xc7, 0x00},
	{HX_CIS_I2C_Action_W, 0xc6, 0x51},
	{HX_CIS_I2C_Action_W, 0xc5, 0x11},
	{HX_CIS_I2C_Action_W, 0xc4, 0x9c},
	{HX_CIS_I2C_Action_W, 0xbf, 0x00},
	{HX_CIS_I2C_Action_W, 0xbc, 0x64},
	{HX_CIS_I2C_Action_W, 0xa6, 0x00},
	{HX_CIS_I2C_Action_W, 0xa7, 0x1e},
	{HX_CIS_I2C_Action_W, 0xa7, 0x6b},
	{HX_CIS_I2C_Action_W, 0xa7, 0x47},
	{HX_CIS_I2C_Action_W, 0xa7, 0x33},
	{HX_CIS_I2C_Action_W, 0xa7, 0x00},
	{HX_CIS_I2C_Action_W, 0xa7, 0x23},
	{HX_CIS_I2C_Action_W, 0xa7, 0x2e},
	{HX_CIS_I2C_Action_W, 0xa7, 0x85},
	{HX_CIS_I2C_Action_W, 0xa7, 0x42},
	{HX_CIS_I2C_Action_W, 0xa7, 0x33},
	{HX_CIS_I2C_Action_W, 0xa7, 0x00},
	{HX_CIS_I2C_Action_W, 0xa7, 0x23},
	{HX_CIS_I2C_Action_W, 0xa7, 0x1b},
	{HX_CIS_I2C_Action_W, 0xa7, 0x74},
	{HX_CIS_I2C_Action_W, 0xa7, 0x42},
	{HX_CIS_I2C_Action_W, 0xa7, 0x33},
	{HX_CIS_I2C_Action_W, 0xa7, 0x00},
	{HX_CIS_I2C_Action_W, 0xa7, 0x23},
	{HX_CIS_I2C_Action_W, 0xc0, 0xc8},
	{HX_CIS_I2C_Action_W, 0xc1, 0x96},
	{HX_CIS_I2C_Action_W, 0x8c, 0x00},
	{HX_CIS_I2C_Action_W, 0x86, 0x3d},
	{HX_CIS_I2C_Action_W, 0x50, 0x92},
	{HX_CIS_I2C_Action_W, 0x51, 0x90},
	{HX_CIS_I2C_Action_W, 0x52, 0x2c},
	{HX_CIS_I2C_Action_W, 0x53, 0x00},
	{HX_CIS_I2C_Action_W, 0x54, 0x00},
	{HX_CIS_I2C_Action_W, 0x55, 0x88},
	{HX_CIS_I2C_Action_W, 0x5a, 0x50},
	{HX_CIS_I2C_Action_W, 0x5b, 0x3c},
	{HX_CIS_I2C_Action_W, 0x5c, 0x00},
	{HX_CIS_I2C_Action_W, 0xd3, 0x04},
	{HX_CIS_I2C_Action_W, 0x7f, 0x00},
	{HX_CIS_I2C_Action_W, 0xda, 0x00},
	{HX_CIS_I2C_Action_W, 0xe5, 0x1f},
	{HX_CIS_I2C_Action_W, 0xe1, 0x67},
	{HX_CIS_I2C_Action_W, 0xe0, 0x00},
	{HX_CIS_I2C_Action_W, 0xdd, 0x7f},
	{HX_CIS_I2C_Action_W, 0x05, 0x00},
	{HX_CIS_I2C_Action_W, 0xff, 0x00},
	{HX_CIS_I2C_Action_W, 0xe0, 0x04},
	{HX_CIS_I2C_Action_W, 0xc0, 0xc8},
	{HX_CIS_I2C_Action_W, 0xc1, 0x96},
	{HX_CIS_I2C_Action_W, 0x86, 0x3d},
	{HX_CIS_I2C_Action_W, 0x50, 0x92},
	{HX_CIS_I2C_Action_W, 0x51, 0x90},
	{HX_CIS_I2C_Action_W, 0x52, 0x2c},
	{HX_CIS_I2C_Action_W, 0x53, 0x00},
	{HX_CIS_I2C_Action_W, 0x54, 0x00},
	{HX_CIS_I2C_Action_W, 0x55, 0x88},
	{HX_CIS_I2C_Action_W, 0x57, 0x00},
	{HX_CIS_I2C_Action_W, 0x5a, 0x50},
	{HX_CIS_I2C_Action_W, 0x5b, 0x3c},
	{HX_CIS_I2C_Action_W, 0x5c, 0x00},
	{HX_CIS_I2C_Action_W, 0xd3, 0x04},
	{HX_CIS_I2C_Action_W, 0xe0, 0x00},
	{HX_CIS_I2C_Action_W, 0xFF, 0x00},
	{HX_CIS_I2C_Action_W, 0x05, 0x00},
	{HX_CIS_I2C_Action_W, 0xda, 0x04},
	{HX_CIS_I2C_Action_W, 0x98, 0x00},
	{HX_CIS_I2C_Action_W, 0x99, 0x00},
	{HX_CIS_I2C_Action_W, 0x00, 0x00},
	{HX_CIS_I2C_Action_W, 0xff, 0x01},
	{HX_CIS_I2C_Action_W, 0x11, 0x00},

	// set the output size of OV2640 to 640*480
	{HX_CIS_I2C_Action_W, 0xff, 0x00},
	{HX_CIS_I2C_Action_W, 0xe0, 0x04},
	{HX_CIS_I2C_Action_W, 0x50, 0x00},
	{HX_CIS_I2C_Action_W, 0x5a, 0x18},
	{HX_CIS_I2C_Action_W, 0x5b, 0x18},
	{HX_CIS_I2C_Action_W, 0x5c, 0x00},
	{HX_CIS_I2C_Action_W, 0xe0, 0x00},
};

static void ov2640_power_init(void)
{
#ifdef CAMERA_ENABLE_PIN
	hx_drv_iomux_set_pmux(OV2640_EN_GPIO, 3);
	hx_drv_iomux_set_outvalue(OV2640_EN_GPIO, OV2640_EN_STATE);
	board_delay_ms(1);
#endif
	return;
}

static void ov2640_power_off(void)
{
#ifdef CAMERA_ENABLE_PIN
	hx_drv_iomux_set_pmux(OV2640_EN_GPIO, 3);
	hx_drv_iomux_set_outvalue(OV2640_EN_GPIO, 1 - OV2640_EN_STATE);
#endif
	return;
}

static int8_t ov2640_get_sensor_id(uint32_t *sensor_id)
{
	uint8_t reg_addr = 0xff;
	uint8_t reg_value = 0x01;

	hx_drv_i2cm_write_data(SS_IIC_2_ID, OV2640_I2C_ADDR, &reg_addr, 1, &reg_value, 1);

	reg_addr = 0x0a;
	if (hx_drv_cis_get_reg(reg_addr, &reg_value) != HX_CIS_NO_ERROR)
	{
		LOGGER_WARNING("<%s><%d>get sensor id failed\n", __func__, __LINE__);
		return -EIO;
	}
	*sensor_id = reg_value << 8;

	reg_addr = 0x0b;
	if (hx_drv_cis_get_reg(reg_addr, &reg_value) != HX_CIS_NO_ERROR)
	{
		LOGGER_WARNING("<%s><%d>get sensor id failed\n", __func__, __LINE__);
		return -EIO;
	}
	*sensor_id = *sensor_id | reg_value;

	return 0;
}

static int8_t ov2640_set_output_size(uint16_t width, uint16_t height)
{
	uint16_t outw, outh;
	uint8_t temp;
	if (width % 4 || height % 4)
	{
		LOGGER_WARNING("camera size param error\n");
		return -EINVAL;
	}
	if (width > OV2640_MAX_WIDTH || height > OV2640_MAX_HEIGHT)
	{
		LOGGER_WARNING("camera size overflow error\n");
		return -EINVAL;
	}

	outw = width / 4;
	outh = height / 4;

	hx_drv_cis_set_reg(0xff, 0x00, 1);
	hx_drv_cis_set_reg(0xe0, 0x04, 1);
	hx_drv_cis_set_reg(0x50, 0x00, 1);
	hx_drv_cis_set_reg(0x5a, outw & 0xff, 1);
	hx_drv_cis_set_reg(0x5b, outh & 0xff, 1);
	temp = (outw >> 8) & 0x03;
	temp |= (outh >> 6) & 0x04;
	hx_drv_cis_set_reg(0x5c, temp, 1);
	hx_drv_cis_set_reg(0xe0, 0X00, 1);

	return 0;
}

Camera_Hal_Struct ov2640_driver = {
	.power_init = ov2640_power_init,
	.power_off = ov2640_power_off,
	.get_sensor_id = ov2640_get_sensor_id,
	.camera_sensor_addr = OV2640_I2C_ADDR,
	.sensor_cfg = sensor_ov2640_setting,
	.sensor_cfg_len = HX_CIS_SIZE_N(sensor_ov2640_setting,
									HX_CIS_SensorSetting_t),
	.xshutdown_pin = CIS_XHSUTDOWN_IOMUX_NONE,
	.xsleep_ctl = SENSORCTRL_XSLEEP_BY_CPU,
	.set_output_size = ov2640_set_output_size,
};
