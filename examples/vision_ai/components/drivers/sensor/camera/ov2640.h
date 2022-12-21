#ifndef OV2640_H
#define OV2640_H

#include "grove_ai_config.h"
#include "camera_core.h"

#ifdef CAMERA_ENABLE_PIN
#define OV2640_EN_GPIO CAMERA_ENABLE_PIN
#define OV2640_EN_STATE CAMERA_ENABLE_STATE
#endif

#define OV2640_I2C_ADDR 0x30

#define OV2640_MAX_WIDTH 1600
#define OV2640_MAX_HEIGHT 1300

#endif
