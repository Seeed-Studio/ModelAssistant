#include "mic_core.h"
#include "error_code.h"
#include "logger.h"

extern Mic_Hal_Struct msm261d_driver;

#if defined(MIC_MSM261D)
Mic_Hal_Struct *mic_hal = &msm261d_driver;
#else
Mic_Hal_Struct *mic_hal = NULL;
#endif

int8_t mic_init(void)
{
    if(mic_hal == NULL){
        LOGGER_WARNING("<%s><%d> mic hal has no implement!!!\n", \
			__func__, __LINE__);
	return -ENODEV;
    }

    mic_hal->init();

    return 0;
}
