#ifndef DATAPATH_H
#define DATAPATH_H

#include "grove_ai_config.h"
#include "logger.h"

#include "hx_drv_hw5x5.h"
#include "hx_drv_jpeg.h"
#include "iot_custom_config.h"

/*Error Retry Count*/
#define MAX_HW5x5JPEG_ERR_RETRY_CNT     10

typedef enum
{
	APP_STATE_INIT,
	APP_STATE_FACE_LIVE_HW5X5JPEG,
	APP_STATE_STOP,
}APP_STATE_E;

typedef struct {
    uint32_t wdma1_startaddr;
    uint32_t wdma2_startaddr;
    uint32_t wdma3_startaddr;
    uint32_t jpegsize_autofill_startaddr;

    uint8_t cyclic_buffer_cnt;
    HW5x5_CFG_T hw5x5_cfg;
    JPEG_CFG_T jpeg_cfg;

    SENSORDPLIB_STREAM_E g_stream_type;

    WE1AppCfg_THROUGH_CV_e through_cv;

    /*
    In order to get a better image. we ignore the 
    first few pictures.
    */
    uint8_t ignore_first_x_pics;
} DataPath_struct;

typedef struct {
    volatile uint8_t g_xdma_abnormal;
    volatile uint8_t g_rs_abnormal;
    volatile uint8_t g_hog_abnormal;
    volatile uint8_t g_rs_frameready;
    volatile uint8_t g_hog_frameready;
    volatile uint8_t g_md_detect;
    volatile uint8_t g_cdm_fifoerror;
    volatile uint8_t g_wdt1_timeout;
    volatile uint8_t g_wdt2_timeout;
    volatile uint8_t g_wdt3_timeout;
    volatile int32_t g_inp1bitparer_abnormal;
    volatile uint32_t g_dp_event;
    volatile uint8_t g_frame_ready;

    volatile uint32_t g_cur_hw5x5jpeg_frame;
    volatile uint32_t g_hw5x5jpeg_acc_frame;

    volatile uint8_t g_frame_process_done;

    volatile APP_STATE_E g_app_cur_state;
    volatile APP_STATE_E g_app_new_state;

    volatile uint32_t g_hw5x5jpeg_err_retry_cnt;

    bool img_ready;

    uint32_t jpeg_enc_addr;
    uint32_t jpeg_enc_filesize;
} DataPath_state;

ERROR_T datapath_init(uint16_t width, uint16_t height);
ERROR_T datapath_start_work(void);
bool datapath_get_img_state();
int datapath_restart();
void datapath_get_jpeg_img(uint32_t *jpeg_enc_addr, uint32_t *jpeg_enc_filesize);
uint32_t datapath_get_yuv_img_addr(void);
ERROR_T datapath_set_roi_start_position(uint16_t start_x, uint16_t start_y);

#endif
