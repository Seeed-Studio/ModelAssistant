#include <string.h>

#include "grove_ai_config.h"
#include "logger.h"

#include "sensor_dp_lib.h"
#include "powermode.h"
#include "hx_drv_inp1bitparser.h"
#include "hx_drv_edm.h"
#include "arc_builtin.h"
#include "board_config.h"
#include "arc.h"
#include "arc_timer.h"

#include "datapath.h"


static DataPath_state datapath_state = {
    0,
};

static DataPath_struct datapath_struct = {
    .wdma1_startaddr = 0x200cbe70,
    .wdma2_startaddr = 0x200f4270,
    .wdma3_startaddr = 0x200ff670,
    .jpegsize_autofill_startaddr = 0x2016fe70,

    .cyclic_buffer_cnt = 1,
    .hw5x5_cfg = {
        .hw5x5_path = HW5x5_PATH_THROUGH_DEMOSAIC,
        .demos_pattern_mode = DEMOS_PATTENMODE_BGGR,
        .demos_color_mode = DEMOS_COLORMODE_YUV422,
        .demos_bndmode = DEMOS_BNDODE_EXTEND0,
        .demoslpf_roundmode = DEMOSLPF_ROUNDMODE_FLOOR,
        .hw55_crop_stx = 0,
        .hw55_crop_sty = 0,
        .hw55_in_width = SENSOR_WIDTH_DEFAULT,
        .hw55_in_height = SENSOR_HEIGHT_DEFAULT,
        .fir_lbp_th = 3,
        .fir_procmode = FIR_PROCMODE_LBP1,
        .firlpf_bndmode = FIRLPF_BNDODE_REPEAT,
    },
    .jpeg_cfg = {
        .jpeg_path = JPEG_PATH_ENCODER_EN,
        .jpeg_enctype = JPEG_ENC_TYPE_YUV422,
        .jpeg_encqtable = JPEG_ENC_QTABLE_4X,
        .enc_width = SENSOR_WIDTH_DEFAULT,
        .enc_height = SENSOR_HEIGHT_DEFAULT,
        .dec_width = SENSOR_WIDTH_DEFAULT,
        .dec_height = SENSOR_HEIGHT_DEFAULT,
        .dec_roi_stx = 0,
        .dec_roi_sty = 0,
    },

    .g_stream_type = SENSORDPLIB_STREAM_NONEAOS,

    .through_cv = WE1AppCfg_THROUGH_CV_YES,
    .ignore_first_x_pics = IGNORE_FIRST_X_PICS,
};

// for debug
#define DBG_APP_PRINT_LEVEL DBG_MORE_INFO

// TODO:
#define SENSOR_STROBE_REQ 0
SENSORDPLIB_HM11B1_HEADER_T info;

// tick cal
volatile uint32_t g_tick_start = 0, g_tick_stop = 0, g_tick_toggle = 0;
volatile uint32_t g_tick_period, g_period;
volatile uint32_t g_tick_sensor_std = 0, g_tick_sensor_stream = 0, g_tick_sensor_toggle = 0;

uint8_t g_jpeg_total_slot = 0;

// struct_algoResult algo_result;

static int app_config_sensor_WE1_rx(uint8_t sensor_init_required, uint8_t sensor_strobe_req)
{
    LOGGER_INFO("<%s><%d>\n", __func__, __LINE__);
    return 0;
}

static void app_iot_facedetect_systemreset()
{
    LOGGER_INFO("\n\n\n\n\nxxxx system reset xxxx\n");
    /*TODO*/
#ifdef EXTERNAL_LDO
    hx_lib_pm_chip_rst(PMU_WE1_POWERPLAN_EXTERNAL_LDO);
#else
    hx_lib_pm_chip_rst(PMU_WE1_POWERPLAN_INTERNAL_LDO);
#endif
}

static void app_sensor_xshutdown_toggle()
{
}

static int app_sensor_standby()
{
    return 0;
}

void app_1bitparser_err_info()
{
    uint32_t de0_count;

    /*get inp1bitparser fsm*/
    hx_drv_inp1bitparser_get_fsm(&info.fsm);
    /*get inp1bitparser HW hvsize*/
    hx_drv_inp1bitparser_get_HW_hvsize(&info.hw_hsize, &info.hw_vsize);
    /*get inp1bitparser hvsize*/
    hx_drv_inp1bitparser_get_hvsize(&info.sensor_hsize, &info.sensor_vsize);
    /*get inp1bitparser frame len, line len*/
    hx_drv_inp1bitparser_get_framelinelen(&info.frame_len, &info.line_len);
    /*get inp1bitparser again*/
    hx_drv_inp1bitparser_get_again(&info.again);
    /*get inp1bitparser dgain*/
    hx_drv_inp1bitparser_get_dgain(&info.dgain);
    /*get inp1bitparser integration time*/
    hx_drv_inp1bitparser_get_intg(&info.intg);
    /*get inp1bitparser interrupt src*/
    hx_drv_inp1bitparser_get_intsrc(&info.intsrc);
    /*get inp1bitparser fstus*/
    hx_drv_inp1bitparser_get_fstus(&info.fstus);
    /*get inp1bitparser fc*/
    hx_drv_inp1bitparser_get_fc(&info.fc);
    /*get inp1bitparser crc*/
    hx_drv_inp1bitparser_get_crc(&info.sensor_crc, &info.hw_crc);
    hx_drv_inp1bitparser_get_cycle(&info.fs_cycle, &info.fe_cycle);
    hx_drv_inp1bitparser_get_fscycle_err_cnt(&info.fs_cycle_err_cnt);
    hx_drv_inp1bitparser_get_errstatus(&info.err_status);

    LOGGER_INFO("fsm=%d\n", info.fsm);
    LOGGER_INFO("hw_hsize=%d,hw_vsize=%d\n", info.hw_hsize, info.hw_vsize);
    LOGGER_INFO("sensor_hsize=%d,sensor_vsize=%d\n", info.sensor_hsize, info.sensor_vsize);
    LOGGER_INFO("sensor_crc=0x%x,hw_crc=0x%x\n", info.sensor_crc, info.hw_crc);
    LOGGER_INFO("fs_cycle=%d,fe_cycle=%d\n", info.fs_cycle, info.fe_cycle);
    LOGGER_INFO("fs_cycle_err_cnt=%d\n", info.fs_cycle_err_cnt);
    LOGGER_INFO("err_status=%d\n", info.err_status);

    hx_drv_inp1bitparser_clear_int();
    hx_drv_inp1bitparser_set_enable(0);
    hx_drv_edm_get_de_count(0, &de0_count);
    LOGGER_INFO("de0_count=%d\n", de0_count);

    sensordplib_stop_capture();
    if (app_sensor_standby() != 0)
    {
        LOGGER_WARNING("standby sensor fail 9\n");
    }
    sensordplib_start_swreset();
    sensordplib_stop_swreset_WoSensorCtrl();
    datapath_state.g_inp1bitparer_abnormal = 0;
    app_sensor_xshutdown_toggle();
    if (app_config_sensor_WE1_rx(1, SENSOR_STROBE_REQ) != 0)
    {
        LOGGER_WARNING("re-setup sensor fail\n");
    }
}

uint8_t app_aiot_face_hw5x5jpeg_algo_check(uint32_t img_addr)
{
    int read_status = 0;
    uint8_t next_jpeg_enc_frameno = 0;
    uint8_t cur_jpeg_enc_frameno = 0;
    uint32_t de0_count;
    uint32_t convde_count;
    uint16_t af_framecnt;
    uint16_t be_framecnt;
    uint8_t wdma1_fin, wdma2_fin, wdma3_fin, rdma_fin;
    uint8_t ready_flag, nframe_end, xdmadone;
    static int poweron_pic_cnt = 0;

    /*Error handling*/
    if (datapath_state.g_xdma_abnormal != 0)
    {
        sensordplib_stop_capture();
        sensordplib_start_swreset();
        sensordplib_stop_swreset_WoSensorCtrl();
        LOGGER_WARNING("abnormal re-setup path cur_frame=%d,\
			acc=%d,event=%d\n",
                       datapath_state.g_cur_hw5x5jpeg_frame,
                       datapath_state.g_hw5x5jpeg_acc_frame,
                       datapath_state.g_dp_event);
        datapath_state.g_xdma_abnormal = 0;
        datapath_state.g_hw5x5jpeg_err_retry_cnt++;
        // need re-setup configuration
        if (datapath_state.g_hw5x5jpeg_err_retry_cnt <
            MAX_HW5x5JPEG_ERR_RETRY_CNT)
        {
            datapath_start_work();
        }
        else
        {
            LOGGER_WARNING("hw5x5jpeg xdma fail overtime\n");
            app_iot_facedetect_systemreset();
        }
    }

    if ((datapath_state.g_wdt1_timeout == 1) || (datapath_state.g_wdt2_timeout == 1) || (datapath_state.g_wdt3_timeout == 1))
    {
        LOGGER_WARNING("EDM WDT timeout event=%d\n", datapath_state.g_dp_event);
        hx_drv_edm_get_de_count(0, &de0_count);
        hx_drv_edm_get_conv_de_count(&convde_count);
        LOGGER_WARNING("de0_count=%d, convde_count=%d\n", de0_count, convde_count);
        sensordplib_get_xdma_fin(&wdma1_fin, &wdma2_fin, &wdma3_fin, &rdma_fin);
        LOGGER_WARNING("wdma1_fin=%d,wdma2_fin=%d,wdma3_fin=%d,rdma_fin=%d\n", wdma1_fin, wdma2_fin, wdma3_fin, rdma_fin);
        sensordplib_get_status(&ready_flag, &nframe_end, &xdmadone);
        LOGGER_WARNING("ready_flag=%d,nframe_end=%d,xdmadone=%d\n", ready_flag, nframe_end, xdmadone);
        hx_drv_edm_get_frame_count(&af_framecnt, &be_framecnt);
        LOGGER_WARNING("af_framecnt=%d,be_framecnt=%d\n", af_framecnt, be_framecnt);

        sensordplib_stop_capture();
        if (app_sensor_standby() != 0)
        {
            LOGGER_WARNING("standby sensor fail 7\n");
        }
        sensordplib_start_swreset();
        sensordplib_stop_swreset_WoSensorCtrl();
        datapath_state.g_wdt1_timeout = 0;
        datapath_state.g_wdt2_timeout = 0;
        datapath_state.g_wdt3_timeout = 0;
        datapath_state.g_hw5x5jpeg_err_retry_cnt++;
        if (datapath_state.g_hw5x5jpeg_err_retry_cnt < MAX_HW5x5JPEG_ERR_RETRY_CNT)
        {
            app_sensor_xshutdown_toggle();
            if (app_config_sensor_WE1_rx(1, SENSOR_STROBE_REQ) != 0)
            {
                LOGGER_WARNING("re-setup sensor fail\n");
            }
            datapath_start_work();
        }
        else
        {
            LOGGER_WARNING("hw5x5jpeg WDT fail overtime\n");
            app_iot_facedetect_systemreset();
        }
    }

    if (datapath_state.g_inp1bitparer_abnormal != 0)
    {
        LOGGER_WARNING("g_inp1bitparer_err=%d\n", datapath_state.g_dp_event);
        datapath_state.g_hw5x5jpeg_err_retry_cnt++;
        datapath_state.g_inp1bitparer_abnormal = 0;
        if (datapath_state.g_hw5x5jpeg_err_retry_cnt < MAX_HW5x5JPEG_ERR_RETRY_CNT)
        {
            app_1bitparser_err_info();
            if (app_config_sensor_WE1_rx(1, SENSOR_STROBE_REQ) != 0)
            {
                LOGGER_WARNING("re-setup sensor fail\n");
            }
            datapath_start_work();
        }
        else
        {
            LOGGER_WARNING("HW5x5JPEG 1bitparser Err retry overtime\n");
            app_iot_facedetect_systemreset();
        }
    }

    // Frame ready
    if (datapath_state.g_frame_ready == 1)
    {
        datapath_state.g_hw5x5jpeg_err_retry_cnt = 0;

        g_tick_stop = _arc_aux_read(AUX_TIMER0_CNT);
        g_tick_period = g_tick_stop - g_tick_start;
        if (is_ref_cpu_clk_by_var())
        {
            g_period = g_tick_period / (get_ref_cpu_clk() / BOARD_SYS_TIMER_HZ);
        }
        else
        {
            g_period = g_tick_period / BOARD_SYS_TIMER_MS_CONV;
        }

        timer_stop(TIMER_0);
        timer_start(TIMER_0, TIMER_CTRL_NH, 0xffffffff); // Set Counter LIMIT to MAX
        g_tick_start = _arc_aux_read(AUX_TIMER0_CNT);
        // LOGGER_INFO("[tick] next frame start:%d \n", g_tick_start);

        // algorithm
        if (datapath_struct.through_cv == WE1AppCfg_THROUGH_CV_YES)
        {
            hx_drv_xdma_get_WDMA2NextFrameIdx(&next_jpeg_enc_frameno);
            hx_drv_xdma_get_WDMA2_bufferNo(&g_jpeg_total_slot);
            if (next_jpeg_enc_frameno == 0)
            {
                cur_jpeg_enc_frameno = g_jpeg_total_slot - 1;
            }
            else
            {
                cur_jpeg_enc_frameno = next_jpeg_enc_frameno - 1;
            }
            hx_drv_jpeg_get_FillFileSizeToMem(cur_jpeg_enc_frameno,
                                              datapath_struct.jpegsize_autofill_startaddr,
                                              &datapath_state.jpeg_enc_filesize);
            hx_drv_jpeg_get_MemAddrByFrameNo(cur_jpeg_enc_frameno,
                                             datapath_struct.wdma2_startaddr,
                                             &datapath_state.jpeg_enc_addr);
            //LOGGER_INFO("next_jpeg_enc_frameno=%d,g_\
			    jpeg_total_slot=%d,jpeg_enc_addr=0x%x \
			    jpeg_enc_filesize=0x%x\n", \
			    next_jpeg_enc_frameno, \
			    g_jpeg_total_slot,datapath_state.jpeg_enc_addr, \
			    datapath_state.jpeg_enc_filesize);

            poweron_pic_cnt++;
            if (poweron_pic_cnt > datapath_struct.ignore_first_x_pics)
            {
                sensordplib_stop_capture();
                datapath_state.img_ready = true;
            }
            //LOGGER_INFO("[cv algo] frame:%d, period:%d (tick)\n", \
			    datapath_state.g_cur_hw5x5jpeg_frame, (tick_algo_stop - tick_algo_start));
        }

        datapath_state.g_frame_ready = 0;

    } // frame ready

    return 0;
}

static void cpu_sleep_at_capture(void)
{
#if 0
    LOGGER_INFO("cpu_sleep_at_capture\n");
    PM_CFG_T aCfg;
    PM_CFG_PWR_MODE_E mode = PM_MODE_CPU_SLEEP;

    hx_lib_get_defcfg_bymode(&aCfg, mode);
    hx_lib_pm_mode_set(aCfg);
#endif
}

static void datapath_nonAOS_restreaming()
{
}

static void datapath_hw5x5jpeg_recapture()
{
    datapath_nonAOS_restreaming();
    sensordplib_retrigger_capture();
    cpu_sleep_at_capture();
}

void datapath_stop_work(void)
{
    sensordplib_stop_capture();
    sensordplib_start_swreset();
    sensordplib_stop_swreset_WoSensorCtrl();
    datapath_state.g_app_cur_state = APP_STATE_STOP;
}

void tinyml_event_handle_callback(SENSORDPLIB_STATUS_E event)
{
    uint8_t human_present = 0;
    uint16_t err;
    datapath_state.g_dp_event = event;

    switch (event)
    {
    case SENSORDPLIB_STATUS_ERR_FS_HVSIZE:
    case SENSORDPLIB_STATUS_ERR_FE_TOGGLE:
    case SENSORDPLIB_STATUS_ERR_FD_TOGGLE:
    case SENSORDPLIB_STATUS_ERR_FS_TOGGLE:
    case SENSORDPLIB_STATUS_ERR_BLANK_ERR: /*reg_inpparser_stall_error*/
    case SENSORDPLIB_STATUS_ERR_CRC_ERR:   /*reg_inpparser_crc_error*/
    case SENSORDPLIB_STATUS_ERR_FE_ERR:    /*reg_inpparser_fe_cycle_error*/
    case SENSORDPLIB_STATUS_ERR_HSIZE_ERR: /*reg_inpparser_hsize_error*/
    case SENSORDPLIB_STATUS_ERR_FS_ERR:    /*reg_inpparser_fs_cycle_error*/
        hx_drv_inp1bitparser_get_errstatus(&err);
        LOGGER_WARNING("err=0x%x\n", err);
        hx_drv_inp1bitparser_clear_int();
        hx_drv_inp1bitparser_set_enable(0);
        datapath_state.g_inp1bitparer_abnormal = 1;
        break;
    case SENSORDPLIB_STATUS_EDM_WDT1_TIMEOUT:
        datapath_state.g_wdt1_timeout = 1;
        break;
    case SENSORDPLIB_STATUS_EDM_WDT2_TIMEOUT:
        datapath_state.g_wdt2_timeout = 1;
        break;
    case SENSORDPLIB_STATUS_EDM_WDT3_TIMEOUT:
        datapath_state.g_wdt3_timeout = 1;
        break;
    case SENSORDPLIB_STATUS_CDM_FIFO_OVERFLOW:
    case SENSORDPLIB_STATUS_CDM_FIFO_UNDERFLOW:
        /*
         * error happen need CDM timing & TPG setting
         * 1. SWRESET Datapath
         * 2. restart streaming flow
         */
        datapath_state.g_cdm_fifoerror = 1;
        break;
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL1:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL2:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL3:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL4:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL5:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL6:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL7:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL8:
    case SENSORDPLIB_STATUS_XDMA_WDMA1_ABNORMAL9:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL1:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL2:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL3:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL4:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL5:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL6:
    case SENSORDPLIB_STATUS_XDMA_WDMA2_ABNORMAL7:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL1:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL2:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL3:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL4:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL5:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL6:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL7:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL8:
    case SENSORDPLIB_STATUS_XDMA_WDMA3_ABNORMAL9:
    case SENSORDPLIB_STATUS_XDMA_RDMA_ABNORMAL1:
    case SENSORDPLIB_STATUS_XDMA_RDMA_ABNORMAL2:
    case SENSORDPLIB_STATUS_XDMA_RDMA_ABNORMAL3:
    case SENSORDPLIB_STATUS_XDMA_RDMA_ABNORMAL4:
    case SENSORDPLIB_STATUS_XDMA_RDMA_ABNORMAL5:
        /*
         * error happen need
         * 1. SWRESET Datapath
         * 2. restart streaming flow
         */
        datapath_state.g_xdma_abnormal = 1;
        break;
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL1:
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL2:
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL3:
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL4:
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL5:
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL6:
    case SENSORDPLIB_STATUS_RSDMA_ABNORMAL7:
        /*
         * error happen need
         * 1. SWRESET RS & RS DMA
         * 2. Re-run flow again
         */
        datapath_state.g_rs_abnormal = 1;
        break;
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL1:
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL2:
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL3:
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL4:
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL5:
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL6:
    case SENSORDPLIB_STATUS_HOGDMA_ABNORMAL7:
        /*
         * error happen need
         * 1. SWRESET HOG & HOG DMA
         * 2. Re-run flow again
         */
        datapath_state.g_hog_abnormal = 1;
        break;
    case SENSORDPLIB_STATUS_CDM_MOTION_DETECT:
        /*
         * app anything want to do
         * */
        datapath_state.g_md_detect = 1;
        break;
    case SENSORDPLIB_STATUS_XDMA_FRAME_READY:
        datapath_state.g_cur_hw5x5jpeg_frame++;
        datapath_state.g_hw5x5jpeg_acc_frame++;

        datapath_state.g_frame_ready = 1;
        break;
    case SENSORDPLIB_STATUS_RSDMA_FINISH:
        datapath_state.g_rs_frameready = 1;
        break;
    case SENSORDPLIB_STATUS_HOGDMA_FINISH:
        datapath_state.g_hog_frameready = 1;
        break;
    case SENSORDPLIB_STATUS_TIMER_FIRE_APP_NOTREADY:
        break;
    default:
        LOGGER_WARNING("Other Event %d\n", event);
        break;
    }

    human_present = app_aiot_face_hw5x5jpeg_algo_check(datapath_struct.wdma3_startaddr);
    datapath_state.g_frame_process_done = 1;

    if (!datapath_state.img_ready)
    {
        datapath_hw5x5jpeg_recapture();
    }
}

ERROR_T datapath_init(uint16_t width, uint16_t height)
{
    sensordplib_start_swreset();
    sensordplib_stop_swreset_WoSensorCtrl();

    sensordplib_set_xDMA_baseaddrbyapp(datapath_struct.wdma1_startaddr,
                                       datapath_struct.wdma2_startaddr,
                                       datapath_struct.wdma3_startaddr);
    sensordplib_set_jpegfilesize_addrbyapp(datapath_struct.jpegsize_autofill_startaddr);

#ifdef EXTERNAL_LDO
    hx_drv_pmu_set_ctrl(PMU_PWR_PLAN, PMU_WE1_POWERPLAN_EXTERNAL_LDO);
    hx_lib_pm_cldo_en(0);
#else
    hx_drv_pmu_set_ctrl(PMU_PWR_PLAN, PMU_WE1_POWERPLAN_INTERNAL_LDO);
#endif

    datapath_struct.hw5x5_cfg.hw55_in_width = width;
    datapath_struct.hw5x5_cfg.hw55_in_height = height;
    datapath_struct.jpeg_cfg.enc_width = width;
    datapath_struct.jpeg_cfg.enc_height = height;
    datapath_struct.jpeg_cfg.dec_width = width;
    datapath_struct.jpeg_cfg.dec_height = height;

    return ERROR_NONE;
}

ERROR_T datapath_start_work(void)
{
    datapath_state.g_app_cur_state = APP_STATE_INIT;
    datapath_state.g_app_new_state = APP_STATE_INIT;

    memset(&datapath_state, 0x00, sizeof(datapath_state));

    datapath_state.g_app_cur_state = APP_STATE_FACE_LIVE_HW5X5JPEG;
    datapath_state.g_app_new_state = APP_STATE_FACE_LIVE_HW5X5JPEG;
    datapath_state.g_frame_process_done = 0;

    sensordplib_set_int_hw5x5_jpeg_wdma23(datapath_struct.hw5x5_cfg,
                                          datapath_struct.jpeg_cfg,
                                          datapath_struct.cyclic_buffer_cnt,
                                          NULL);

    hx_dplib_register_cb(tinyml_event_handle_callback, SENSORDPLIB_CB_FUNTYPE_DP);

    sensordplib_set_sensorctrl_start();

    cpu_sleep_at_capture();

    return ERROR_NONE;
}

bool datapath_get_img_state()
{
    return datapath_state.img_ready;
}

int datapath_restart()
{
    int ret = 0;
    /*
    we set the img_ready to false to enable a new capture
    */
    datapath_state.img_ready = false;

    sensordplib_set_int_hw5x5_jpeg_wdma23(datapath_struct.hw5x5_cfg,
                                          datapath_struct.jpeg_cfg,
                                          datapath_struct.cyclic_buffer_cnt,
                                          NULL);
    ret = sensordplib_set_sensorctrl_start();

    return ret;
}

void datapath_get_jpeg_img(uint32_t *jpeg_enc_addr, uint32_t *jpeg_enc_filesize)
{
    *jpeg_enc_addr = datapath_state.jpeg_enc_addr;
    *jpeg_enc_filesize = datapath_state.jpeg_enc_filesize;
}

uint32_t datapath_get_yuv_img_addr(void)
{
    return datapath_struct.wdma3_startaddr;
}

ERROR_T datapath_set_roi_start_position(uint16_t start_x, uint16_t start_y)
{
    datapath_struct.hw5x5_cfg.hw55_crop_stx = start_x;
    datapath_struct.hw5x5_cfg.hw55_crop_sty = start_y;

    return ERROR_NONE;
}
