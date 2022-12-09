#include "hx_drv_webusb.h"
#include "hx_drv_iomux.h"
#include "hx_drv_uart.h"
#include "console_io.h"
#include "debugger.h"

#define WEBUSB_SYNC_PIN IOMUX_PGPIO8
#define WEBUSB_SYNC_STATE 1

static DEV_UART *console_uart;

void hx_drv_webusb_init()
{
    hx_drv_iomux_set_pmux(WEBUSB_SYNC_PIN, 3);
    hx_drv_iomux_set_outvalue(WEBUSB_SYNC_PIN, 1 - WEBUSB_SYNC_STATE);
    console_uart = hx_drv_uart_get_dev(CONSOLE_UART_ID);
}

void hx_drv_webusb_write_vision(uint8_t *data, uint32_t size)
{
    if (console_uart == NULL || data == NULL)
    {
        return;
    }
    if (!debugger_available())
    {
        return;
    }
    board_delay_ms(1);
    hx_drv_iomux_set_outvalue(WEBUSB_SYNC_PIN, WEBUSB_SYNC_STATE);
    board_delay_ms(1);
    uint8_t image_header[8] = {0};
    image_header[0] = (WEBUSB_PROTOCOL_VISION_MAGIC & 0xFF000000) >> 24;
    image_header[1] = (WEBUSB_PROTOCOL_VISION_MAGIC & 0xFF0000) >> 16;
    image_header[2] = (WEBUSB_PROTOCOL_VISION_MAGIC & 0xFF00) >> 8;
    image_header[3] = (WEBUSB_PROTOCOL_VISION_MAGIC & 0xFF);
    image_header[4] = (size & 0xFF000000) >> 24;
    image_header[5] = (size & 0xFF0000) >> 16;
    image_header[6] = (size & 0xFF00) >> 8;
    image_header[7] = (size & 0xFF);

    console_uart->uart_write(image_header, 8);

    board_delay_ms(1);
    console_uart->uart_write(data, size);
    board_delay_ms(1);
    hx_drv_iomux_set_outvalue(WEBUSB_SYNC_PIN, 1 - WEBUSB_SYNC_STATE);
}

void hx_drv_webusb_write_text(uint8_t *data, uint32_t size)
{
    if (console_uart == NULL || data == NULL)
    {
        return;
    }
    if (!debugger_available())
    {
        return;
    }
    board_delay_ms(1);
    hx_drv_iomux_set_outvalue(WEBUSB_SYNC_PIN, WEBUSB_SYNC_STATE);
    board_delay_ms(1);
    uint8_t image_header[8] = {0};
    image_header[0] = (WEBUSB_PROTOCOL_TEXT_MAGIC & 0xFF000000) >> 24;
    image_header[1] = (WEBUSB_PROTOCOL_TEXT_MAGIC & 0xFF0000) >> 16;
    image_header[2] = (WEBUSB_PROTOCOL_TEXT_MAGIC & 0xFF00) >> 8;
    image_header[3] = (WEBUSB_PROTOCOL_TEXT_MAGIC & 0xFF);
    image_header[4] = (size & 0xFF000000) >> 24;
    image_header[5] = (size & 0xFF0000) >> 16;
    image_header[6] = (size & 0xFF00) >> 8;
    image_header[7] = (size & 0xFF);

    console_uart->uart_write(image_header, 8);

    board_delay_ms(1);
    console_uart->uart_write(data, size);
    board_delay_ms(1);
    hx_drv_iomux_set_outvalue(WEBUSB_SYNC_PIN, 1 - WEBUSB_SYNC_STATE);
}


