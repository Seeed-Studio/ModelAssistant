#ifndef HX_DRV_WEBUSB_H
#define HX_DRV_WEBUSB_H

#include <stdint.h>

#include "grove_ai_config.h"
#include "logger.h"

#include "hx_drv_iomux.h"

#define WEBUSB_PROTOCOL_VISION_MAGIC 0x2B2D2B2D
#define WEBUSB_PROTOCOL_AUDIO_MAGIC 0x2E2A2E2A
#define WEBUSB_PROTOCOL_AXIS_MAGIC 0x2C2F2C2F
#define WEBUSB_PROTOCOL_TEXT_MAGIC 0x0F100E12

#ifdef __cplusplus
extern "C"
{
#endif

    void hx_drv_webusb_init();

    void hx_drv_webusb_write_vision(uint8_t *jpeg, uint32_t length);
    void hx_drv_webusb_write_text(uint8_t *data, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif