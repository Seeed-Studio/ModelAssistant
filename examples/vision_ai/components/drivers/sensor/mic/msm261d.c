#include "msm261d.h"
#include "error_code.h"
#include "hx_drv_timer.h"
#include "logger.h"

static audio_config_t app_air_aud_cfg;

static uint8_t audio_buf[4096*10] = {0,};

static int pdm_cnt = 0;
static bool pdm_flag = false;

// raise audio rx event
static void audio_rx_callback_fun(uint32_t status)
{
    uint32_t audio_buf_addr;
    uint32_t block;
    uint16_t temp;
    uint32_t temp1;

    hx_lib_audio_request_read(&audio_buf_addr, &block);

    for(int i = 0;i < 1024; i = i + 2){
        temp = *((uint16_t *)audio_buf_addr+i);
        LOGGER_INFO("%04x\n", temp);
    }

    hx_lib_audio_update_idx(&block);

    return ;
}

static int8_t msm261d_init(void)
{
    uint32_t temp;
    AUDIO_ERROR_E ret;
    hx_lib_audio_set_if(AUDIO_IF_PDM);
    hx_lib_audio_init();
    // register the callback function to support event handler
    hx_lib_audio_register_evt_cb(audio_rx_callback_fun);
    app_air_aud_cfg.sample_rate = AUDIO_SR_16KHZ;
    app_air_aud_cfg.buffer_addr = audio_buf;//malloc(4096 * 10); // (uint32_t *) (0x20000000+0xB000);//44*1024
    app_air_aud_cfg.block_num = 10;
    app_air_aud_cfg.block_sz = 4096;
    app_air_aud_cfg.cb_evt_blk = 1;
    ret = hx_lib_audio_start(&app_air_aud_cfg);
    LOGGER_INFO("msm261d pdm init : %d\n", ret);

    return 0;
}

Mic_Hal_Struct msm261d_driver = {
    .init = msm261d_init,
};
