#include "communication_core.h"
#include "hx_drv_webusb.h"

int8_t  communication_init(void)
{
#if defined(USE_WEBUSB)
    hx_drv_webusb_init();
#endif
    return 0;
}
