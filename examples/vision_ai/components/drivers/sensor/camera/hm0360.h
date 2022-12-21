#ifndef HM0360_H
#define HM0360_H

#include "camera_core.h"

#ifdef CAMERA_ENABLE_PIN
#define HM0360_EN_GPIO CAMERA_ENABLE_PIN
#define HM0360_EN_STATE CAMERA_ENABLE_STATE
#else
#define HM0360_EN_GPIO IOMUX_RESERVED
#define HM0360_EN_STATE 0
#endif

#define HM0360_I2C_ADDR 0x24

#endif
