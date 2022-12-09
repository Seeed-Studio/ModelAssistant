#include <stdlib.h>
#include <string.h>

#include "hx_drv_lsm6ds3.h"
#include "hx_drv_iic_m.h"

#include "grove_ai_config.h"
#include "logger.h"

static DEV_LSM6DS3 _dev_lsm6ds3;

static int8_t _read_register_region(DEV_LSM6DS3_PTR lsm6ds3, uint8_t *out, uint8_t offset, uint8_t length)
{
    if (lsm6ds3 == NULL || out == NULL)
    {
        return -1;
    }
    if (hx_drv_i2cm_writeread(lsm6ds3->i2c, lsm6ds3->address, &offset, 1, out, length) < 0)
    {
        memset(out, 0, length);
        return -1;
    }
    return 0;
}

static int8_t _read_register(DEV_LSM6DS3_PTR lsm6ds3, uint8_t *out, uint8_t offset)
{
    return _read_register_region(lsm6ds3, out, offset, 1);
}

static int8_t _read_register_int16(DEV_LSM6DS3_PTR lsm6ds3, int16_t *out, uint8_t offset)
{
    uint8_t buf[2];
    _read_register_region(lsm6ds3, buf, offset, 2);
    int16_t value = (int16_t)buf[0] | (int16_t)(buf[1] << 8);
    *out = value;
    return 0;
}

static int8_t _write_register_region(DEV_LSM6DS3_PTR lsm6ds3, uint8_t offset, uint8_t *data, uint8_t length)
{
    if (lsm6ds3 == NULL || data == NULL)
    {
        return -1;
    }

    if (hx_drv_i2cm_write_data(lsm6ds3->i2c, lsm6ds3->address, &offset, 1, data, length) < 0)
    {
        return -1;
    }
    return 0;
}

static int8_t _write_register(DEV_LSM6DS3_PTR lsm6ds3, uint8_t offset, uint8_t value)
{

    return _write_register_region(lsm6ds3, offset, &value, 1);
}

DEV_LSM6DS3_PTR hx_drv_lsm6ds3_init(USE_SS_IIC_E i2c)
{
    DEV_LSM6DS3_PTR lsm6ds3 = (DEV_LSM6DS3_PTR)&_dev_lsm6ds3; // malloc(sizeof(DEV_LSM6DS3));

    lsm6ds3->address = LSM6DS3_ACC_ADDR;
    lsm6ds3->i2c = i2c;
    hx_drv_i2cm_init(lsm6ds3->i2c, IIC_SPEED_STANDARD);

    lsm6ds3->settings.gyro_enabled = 1;         // Can be 0 or 1
    lsm6ds3->settings.gyro_range = 2000;        // Max deg/s.  Can be: 125, 245, 500, 1000, 2000
    lsm6ds3->settings.gyro_sample_rate = 416;   // Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666
    lsm6ds3->settings.gyro_band_width = 400;    // Hz.  Can be: 50, 100, 200, 400;
    lsm6ds3->settings.gyro_fifo_enabled = 1;    // Set to include gyro in FIFO
    lsm6ds3->settings.gyro_fifo_decimation = 1; // set 1 for on /1

    lsm6ds3->settings.accel_enabled = 1;
    lsm6ds3->settings.accel_odr_off = 1;
    lsm6ds3->settings.accel_range = 16;          // Max G force readable.  Can be: 2, 4, 8, 16
    lsm6ds3->settings.accel_sample_rate = 416;   // Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666, 3332, 6664, 13330
    lsm6ds3->settings.accel_band_width = 100;    // Hz.  Can be: 50, 100, 200, 400;
    lsm6ds3->settings.accel_fifo_enabled = 1;    // Set to include accelerometer in the FIFO
    lsm6ds3->settings.accel_fifo_decimation = 1; // set 1 for on /1

    lsm6ds3->settings.temp_enabled = 1;

    // Select interface mode
    lsm6ds3->settings.comm_mode = 1; // Can be modes 1, 2 or 3

    // FIFO control data
    lsm6ds3->settings.fifo_threshold = 3000; // Can be 0 to 4096 (16 bit bytes)
    lsm6ds3->settings.fifo_sample_rate = 10; // default 10Hz
    lsm6ds3->settings.fifo_mode_word = 0;    // Default off

    // Return WHO AM I reg  //Not no mo!
    uint8_t result;
    _read_register(lsm6ds3, &result, LSM6DS3_ACC_GYRO_WHO_AM_I_REG);

    if (result == LSM6DS3_ACC_GYRO_WHO_AM_I)
    {                                            // 0x69 LSM6DS3
        lsm6ds3->settings.temp_sensitivity = 16; // Sensitivity to scale 16
    }
    else if (result == LSM6DS3_C_ACC_GYRO_WHO_AM_I)
    {                                             // 0x6A LSM6dS3-C
        lsm6ds3->settings.temp_sensitivity = 256; // Sensitivity to scale 256
    }
    else
    {
        free(lsm6ds3);
        lsm6ds3 = NULL;
    }

    return lsm6ds3;
}

bool hx_drv_lsm6ds3_begin(DEV_LSM6DS3_PTR lsm6ds3)
{
    // Setup the accelerometer******************************
    uint8_t value = 0; // Start Fresh!
    if (lsm6ds3->settings.accel_enabled == 1)
    {
        // Build config reg
        // First patch in filter bandwidth
        switch (lsm6ds3->settings.accel_band_width)
        {
        case 50:
            value |= LSM6DS3_ACC_GYRO_BW_XL_50Hz;
            break;
        case 100:
            value |= LSM6DS3_ACC_GYRO_BW_XL_100Hz;
            break;
        case 200:
            value |= LSM6DS3_ACC_GYRO_BW_XL_200Hz;
            break;
        default: // set default case to max passthrough
        case 400:
            value |= LSM6DS3_ACC_GYRO_BW_XL_400Hz;
            break;
        }
        // Next, patch in full scale
        switch (lsm6ds3->settings.accel_range)
        {
        case 2:
            value |= LSM6DS3_ACC_GYRO_FS_XL_2g;
            break;
        case 4:
            value |= LSM6DS3_ACC_GYRO_FS_XL_4g;
            break;
        case 8:
            value |= LSM6DS3_ACC_GYRO_FS_XL_8g;
            break;
        default: // set default case to 16(max)
        case 16:
            value |= LSM6DS3_ACC_GYRO_FS_XL_16g;
            break;
        }
        // Lastly, patch in accelerometer ODR
        switch (lsm6ds3->settings.accel_sample_rate)
        {
        case 13:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_13Hz;
            break;
        case 26:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_26Hz;
            break;
        case 52:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_52Hz;
            break;
        default: // Set default to 104
        case 104:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_104Hz;
            break;
        case 208:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_208Hz;
            break;
        case 416:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_416Hz;
            break;
        case 833:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_833Hz;
            break;
        case 1660:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_1660Hz;
            break;
        case 3330:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_3330Hz;
            break;
        case 6660:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_6660Hz;
            break;
        case 13330:
            value |= LSM6DS3_ACC_GYRO_ODR_XL_13330Hz;
            break;
        }
    }
    else
    {
        value = 0;
    }

    // Now, write the patched together data
    _write_register(lsm6ds3, LSM6DS3_ACC_GYRO_CTRL1_XL, value);

    // Set the ODR bit
    value = 0;
    _read_register(lsm6ds3, &value, LSM6DS3_ACC_GYRO_CTRL4_C);
    value &= ~((uint8_t)LSM6DS3_ACC_GYRO_BW_SCAL_ODR_ENABLED);
    if (lsm6ds3->settings.accel_odr_off == 1)
    {
        value |= LSM6DS3_ACC_GYRO_BW_SCAL_ODR_ENABLED;
    }
    _write_register(lsm6ds3, LSM6DS3_ACC_GYRO_CTRL4_C, value);

    // Setup the gyroscope**********************************************
    value = 0; // Start Fresh!
    if (lsm6ds3->settings.gyro_enabled == 1)
    {
        // Build config reg
        // First, patch in full scale
        switch (lsm6ds3->settings.gyro_range)
        {
        case 125:
            value |= LSM6DS3_ACC_GYRO_FS_125_ENABLED;
            break;
        case 245:
            value |= LSM6DS3_ACC_GYRO_FS_G_245dps;
            break;
        case 500:
            value |= LSM6DS3_ACC_GYRO_FS_G_500dps;
            break;
        case 1000:
            value |= LSM6DS3_ACC_GYRO_FS_G_1000dps;
            break;
        default: // Default to full 2000DPS range
        case 2000:
            value |= LSM6DS3_ACC_GYRO_FS_G_2000dps;
            break;
        }
        // Lastly, patch in gyro ODR
        switch (lsm6ds3->settings.gyro_sample_rate)
        {
        case 13:
            value |= LSM6DS3_ACC_GYRO_ODR_G_13Hz;
            break;
        case 26:
            value |= LSM6DS3_ACC_GYRO_ODR_G_26Hz;
            break;
        case 52:
            value |= LSM6DS3_ACC_GYRO_ODR_G_52Hz;
            break;
        default: // Set default to 104
        case 104:
            value |= LSM6DS3_ACC_GYRO_ODR_G_104Hz;
            break;
        case 208:
            value |= LSM6DS3_ACC_GYRO_ODR_G_208Hz;
            break;
        case 416:
            value |= LSM6DS3_ACC_GYRO_ODR_G_416Hz;
            break;
        case 833:
            value |= LSM6DS3_ACC_GYRO_ODR_G_833Hz;
            break;
        case 1660:
            value |= LSM6DS3_ACC_GYRO_ODR_G_1660Hz;
            break;
        }
    }
    else
    {
        value = 0;
    }
    // Write the byte
    _write_register(lsm6ds3, LSM6DS3_ACC_GYRO_CTRL2_G, value);

    return true;
}
uint8_t hx_drv_lsm6ds3_gyro_available(DEV_LSM6DS3_PTR lsm6ds3)
{
    uint8_t output;
    _read_register(lsm6ds3, &output, LSM6DS3_ACC_GYRO_STATUS_REG);
    return output & 0x01;
}
uint8_t hx_drv_lsm6ds3_acc_available(DEV_LSM6DS3_PTR lsm6ds3)
{
    uint8_t output;
    _read_register(lsm6ds3, &output, LSM6DS3_ACC_GYRO_STATUS_REG);
    return output & 0x02;
}

int16_t hx_drv_lsm6ds3_read_raw_temp(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUT_TEMP_L);
    return output;
}

int16_t hx_drv_lsm6ds3_read_raw_acc_x(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUTX_L_XL);
    return output;
}
int16_t hx_drv_lsm6ds3_read_raw_acc_y(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUTY_L_XL);
    return output;
}
int16_t hx_drv_lsm6ds3_read_raw_acc_z(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUTZ_L_XL);
    return output;
}
int16_t hx_drv_lsm6ds3_read_raw_gyro_x(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUTX_L_G);
    return output;
}
int16_t hx_drv_lsm6ds3_read_raw_gyro_y(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUTY_L_G);
    return output;
}
int16_t hx_drv_lsm6ds3_read_raw_gyro_z(DEV_LSM6DS3_PTR lsm6ds3)
{
    int16_t output;
    _read_register_int16(lsm6ds3, &output, LSM6DS3_ACC_GYRO_OUTZ_L_G);
    return output;
}

float hx_drv_lsm6ds3_read_temp_c(DEV_LSM6DS3_PTR lsm6ds3)
{
    float output = (float)hx_drv_lsm6ds3_read_raw_temp(lsm6ds3) / lsm6ds3->settings.temp_sensitivity;

    output += 25; // Add 25 degrees to remove offset

    return output;
}
float hx_drv_lsm6ds3_read_temp_f(DEV_LSM6DS3_PTR lsm6ds3)
{
    float output = (float)hx_drv_lsm6ds3_read_raw_temp(lsm6ds3) / lsm6ds3->settings.temp_sensitivity;
    output += 25; // Add 25 degrees to remove offsetv
    output = (output * 9) / 5 + 32;

    return output;
}
float hx_drv_lsm6ds3_read_acc_x(DEV_LSM6DS3_PTR lsm6ds3)
{
    float output = (float)hx_drv_lsm6ds3_read_raw_acc_x(lsm6ds3) * 0.061 * (lsm6ds3->settings.accel_range >> 1) / 1000;
    return output;
}
float hx_drv_lsm6ds3_read_acc_y(DEV_LSM6DS3_PTR lsm6ds3)
{
    float output = (float)hx_drv_lsm6ds3_read_raw_acc_y(lsm6ds3) * 0.061 * (lsm6ds3->settings.accel_range >> 1) / 1000;
    return output;
}
float hx_drv_lsm6ds3_read_acc_z(DEV_LSM6DS3_PTR lsm6ds3)
{
    float output = (float)hx_drv_lsm6ds3_read_raw_acc_z(lsm6ds3) * 0.061 * (lsm6ds3->settings.accel_range >> 1) / 1000;
    return output;
}
float hx_drv_lsm6ds3_read_gyro_x(DEV_LSM6DS3_PTR lsm6ds3)
{
    uint8_t gyro_range_divisor = lsm6ds3->settings.gyro_range / 125;
    if (lsm6ds3->settings.gyro_range == 245)
    {
        gyro_range_divisor = 2;
    }

    float output = (float)hx_drv_lsm6ds3_read_raw_gyro_x(lsm6ds3) * 4.375 * (gyro_range_divisor) / 1000;
    return output;
}
float hx_drv_lsm6ds3_read_gyro_y(DEV_LSM6DS3_PTR lsm6ds3)
{
    uint8_t gyro_range_divisor = lsm6ds3->settings.gyro_range / 125;
    if (lsm6ds3->settings.gyro_range == 245)
    {
        gyro_range_divisor = 2;
    }
    float output = (float)hx_drv_lsm6ds3_read_raw_gyro_y(lsm6ds3) * 4.375 * (gyro_range_divisor) / 1000;
    return output;
}
float hx_drv_lsm6ds3_read_gyro_z(DEV_LSM6DS3_PTR lsm6ds3)
{
    uint8_t gyro_range_divisor = lsm6ds3->settings.gyro_range / 125;
    if (lsm6ds3->settings.gyro_range == 245)
    {
        gyro_range_divisor = 2;
    }
    float output = (float)hx_drv_lsm6ds3_read_raw_gyro_z(lsm6ds3) * 4.375 * (gyro_range_divisor) / 1000;
    return output;
}