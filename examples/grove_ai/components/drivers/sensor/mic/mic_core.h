#ifndef MIC_CORE_H
#define MIC_CORE_H

#include "grove_ai_config.h"
#include <stdio.h>
#include "aud_lib.h"

//include different camera driver
#include "msm261d.h"

typedef struct {
    int8_t (*init)(void);
} Mic_Hal_Struct;

int8_t mic_init(void);

#endif
