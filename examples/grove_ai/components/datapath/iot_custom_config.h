/*
 * application_config.h
 *
 *  Created on: 2019¦~7¤ë4¤é
 *      Author: 902447
 */

#ifndef SCENARIO_APP_IOT_CUSTOM_CONFIG_H_
#define SCENARIO_APP_IOT_CUSTOM_CONFIG_H_

#include "sensor_dp_lib.h"
#include "hx_drv_CIS_common.h"
#include "hx_drv_adcc.h"
#include "powermode.h"

#define APP_CONFIG_TABLE_VERSION  0x0001

#define MAX_SUPPORT_SENSOR_CFG_COUNT   	700
#define MAX_SENSOR_STREAM_CFG_COUNT   	5
#define MAX_SENSOR_STROBE_CFG_COUNT		5
#define MAX_SUPPORT_WAKEUP_CPU_INT_PIN  9
/**
 * \enum WE1AppCfg_TableType_e
 * \brief WE1 Configuration table type
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_TableType_APP = 0,    		/**< Table Type - Application level */
	WE1AppCfg_TableType_SENSOR_CFG,        /**< Table Type - Sensor configuration */
	WE1AppCfg_TableType_SENSOR_STREAM_ON,  /**< Table Type - Sensor Stream On */
	WE1AppCfg_TableType_SENSOR_STREAM_OFF, /**< Table Type - Sensor Stream Off */
	WE1AppCfg_TableType_WE1_DRIVER,        /**< Table Type - WE-1 Driver */
	WE1AppCfg_TableType_SOC_COMMUNICATION, /**< Table Type - SOC Communication */
}WE1AppCfg_TableType_e;

/**
 * \enum WE1AppCfg_APPType_e
 * \brief WE1 Application type
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_APPType_HUMANDET_CDM = 0,    		/**< App Type - Human detection sensor, Always-on camera (CDM Mode)*/
	WE1AppCfg_APPType_HUMANDET_GPIO_WAKEUP,        /**< App Type - Human Presence sensor, GPIO activated */
	WE1AppCfg_APPType_HUMANDET_ANALOG_DEVICE,  	/**< App Type - Human Presence sensor, Analog Device activated*/
	WE1AppCfg_APPType_OCCUPANCY_SENSOR, 		/**< App Type - Occupancy Sensor */
	WE1AppCfg_APPType_FACE_DETECT_ALLON,        		/**< App Type - Face Detect */
	WE1AppCfg_APPType_FACE_DETECT_LOWPOWER,        		/**< App Type - Face Detect with CDM*/
	WE1AppCfg_APPType_HUMANDETECT_SIMPLE_LOWPOWER,        		/**< App Type - Human Detect with CDM*/
	WE1AppCfg_APPType_ALGODET_SIMPLE_ANALOG_DEVICE,  	/**< App Type - Algo Detect, Analog PIR activated 1 JPEG*/
	WE1AppCfg_APPType_ALGODET_SENSORMD_WAKEUP,  	/**< App Type - Algo Detect, Sensor MD activated 1 JPEG*/
	WE1AppCfg_APPType_ALGODET_CDM_ANALOGE_WAKEUP,  	/**< App Type - Algo Detect, CDM, GPIO, Analog Deive activated 1 JPEG*/
	WE1AppCfg_APPType_ALGODET_PREROLLING_GPIOWAKEUP,  	/**< App Type - Algo Detect, PreRolling GPIO Wakeup*/
	WE1AppCfg_APPType_ALGODET_PERODICAL_WAKEUP_QUICKBOOT,  	/**< App Type - Algo Detect, Periodical Wakeup Quick Boot*/
}WE1AppCfg_APPType_e;

/**
 * \enum WE1AppCfg_AlgoDetect_Monitor_e
 * \brief WE1 Algo Detect type
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_AlgoDetect_Monitor_BYFRAMENO = 0,    /**< Keep monitor Stop by Continuing Not detect frame Number*/
	WE1AppCfg_AlgoDetect_Monitor_BYDETECT,        /**< Stop monitor by No Detect*/
	WE1AppCfg_AlgoDetect_NOT_Monitor,        /**< Detect then go back to CDM or PIR*/
}WE1AppCfg_AlgoDetect_Monitor_e;


/**
 * \enum WE1AppCfg_SOCCOMType_e
 * \brief WE1 SOC Communication type
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_SOCCOMType_I2C_SLAVE = 0,    /**< SOC Communication type - I2C Slave*/
	WE1AppCfg_SOCCOMType_SPI_SLAVE,        /**< SOC Communication type - SPI Slave*/
	WE1AppCfg_SOCCOMType_SPI_MASTER,  		/**< SOC Communication type - SPI Master*/
	WE1AppCfg_SOCCOMType_UART, 			/**< SOC Communication type - UART*/
}WE1AppCfg_SOCCOMType_e;

/**
 * \enum WE1AppCfg_THROUGH_CV_e
 * \brief WE1 Through CV or not
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_THROUGH_CV_NO = 0,    /**< Not Through CV*/
	WE1AppCfg_THROUGH_CV_YES,        /**< Through CV*/
}WE1AppCfg_THROUGH_CV_e;

/**
 * \enum WE1AppCfg_GPIO_e
 * \brief WE1 GPIO Selection
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_GPIO_IOMUX_PGPIO0 = 0,
	WE1AppCfg_GPIO_IOMUX_PGPIO1,
	WE1AppCfg_GPIO_IOMUX_PGPIO2,
	WE1AppCfg_GPIO_IOMUX_PGPIO3,
	WE1AppCfg_GPIO_IOMUX_PGPIO4,
	WE1AppCfg_GPIO_IOMUX_PGPIO5,
	WE1AppCfg_GPIO_IOMUX_PGPIO6,
	WE1AppCfg_GPIO_IOMUX_PGPIO7,
	WE1AppCfg_GPIO_IOMUX_PGPIO8,
	WE1AppCfg_GPIO_IOMUX_PGPIO9,
	WE1AppCfg_GPIO_IOMUX_PGPIO10,
	WE1AppCfg_GPIO_IOMUX_PGPIO11,
	WE1AppCfg_GPIO_IOMUX_PGPIO12,
	WE1AppCfg_GPIO_IOMUX_PGPIO13,
	WE1AppCfg_GPIO_IOMUX_PGPIO14,
	WE1AppCfg_GPIO_IOMUX_RESERVED,	// reserved
	WE1AppCfg_GPIO_IOMUX_SGPIO0 = 16,
	WE1AppCfg_GPIO_IOMUX_SGPIO1,
	WE1AppCfg_GPIO_IOMUX_NONE
}WE1AppCfg_GPIO_e;

/**
 * \enum WE1AppCfg_Ambient_Light_e
 * \brief WE1 Ambient Light support or not
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_Ambient_Light_NO = 0,    /**< Ambient Light Not support*/
	WE1AppCfg_Ambient_Light_Support,    /**< Ambient Light support*/
}WE1AppCfg_Ambient_Light_e;

/**
 * \enum WE1AppCfg_PDM_Support_e
 * \brief WE1 PDM Support or not
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_PDM_Support_NO = 0,    /**< PDM Not Support*/
	WE1AppCfg_PDM_Support_YES,        /**< PDM Support*/
}WE1AppCfg_PDM_Support_e;

/**
 * \enum WE1AppCfg_I2S_Support_e
 * \brief WE1 I2S Support or not
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_I2S_Support_NO = 0,    /**< PDM Not Support*/
	WE1AppCfg_I2S_Support_YES,        /**< PDM Support*/
}WE1AppCfg_I2S_Support_e;

/**
 * \enum WE1AppCfg_CHIP_Package_e
 * \brief WE1 Chip Package
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_CHIP_Package_LQFP128 = 0,    /**< WE-1 Chip LQFP128 */
	WE1AppCfg_CHIP_Package_WLCSP38,        /**< WE-1 Chip WLCSP38 */
}WE1AppCfg_CHIP_Package_e;

/**
 * \enum WE1AppCfg_DP_CLK_Mux_e
 * \brief WE1 DP CLK Mux
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_DP_CLK_Mux_RC36M = 0,    /**< DP Clock from RC36M*/
	WE1AppCfg_DP_CLK_Mux_XTAL,        /**< DP Clock from XTAL*/
}WE1AppCfg_DP_CLK_Mux_e;

/**
 * \enum WE1AppCfg_MCLK_CLK_Mux_e
 * \brief WE1 MCLK CLK Mux
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_MCLK_CLK_Mux_RC36M = 0,    /**< MCLK Clock from RC36M*/
	WE1AppCfg_MCLK_CLK_Mux_XTAL,        /**< MCLK Clock from XTAL*/
}WE1AppCfg_MCLK_CLK_Mux_e;

/**
 * \enum WE1AppCfg_SensorColorType_e
 * \brief WE1 Sensor Color type
 */
typedef enum EMBARC_PACKED
{
	WE1AppCfg_SensorColorType_MONO 		= 0,    	/**< Sensor Color Type - Mono*/
	WE1AppCfg_SensorColorType_BAYER 		= 1,        /**< Sensor Color Type - Bayer*/
}WE1AppCfg_SensorColorType_e;

typedef struct
{
	uint16_t	   table_version; 	/**< Table Version */
	uint16_t	   totalLen; 	    /**< Table Total Len */
	uint16_t       table_Len;       /**< sizeof(WE1AppCfg_TableInfo_t)*u32CateCount + sizeof(WE1AppCfg_TableHeader_t) */
	uint8_t		   table_cate_count;/**< category count*/
	uint8_t		   table_crc;/**< Table CRC*/
	uint16_t	   table_checksum; 	/**< checksum calculate from table header not include checksum */
}EMBARC_PACKED WE1AppCfg_TableHeader_t;

typedef struct
{
	WE1AppCfg_TableType_e table_type;
	uint16_t u16Offset;
	uint16_t u16Len;
}EMBARC_PACKED WE1AppCfg_TableInfo_t;


typedef struct
{
	WE1AppCfg_APPType_e app_type;  /**< Application Type */
    uint32_t pmu_sensor_rtc; /**< PMU Mode RTC Interval */
    uint32_t pmu_sensor_wdg; /**< PMU Mode Sensor WDG */
    uint32_t classification_rtc; /**< Classification Mode RTC Interval if the value is 0, it continues do classification*/
    uint32_t adc_rtc;/**< ADC RTC Interval */
    uint32_t adc_sample_period;/**< ADC Sample Period */
    uint16_t classification_detect_max_frame;/**< Classification Detection Frame max numbers for no algo detect*/
    WE1AppCfg_AlgoDetect_Monitor_e detect_monitor_mode;/**< After Algo detect, monitor stop method*/
    uint16_t nodetect_monitor_frame;/**< After Algo detect, monitor stop by N frame no object detect*/
    uint8_t cyclic_check_frame;/**< No Human detect in live capture, jpeg cyclic buffer check frame no*/
    WE1AppCfg_THROUGH_CV_e through_cv; /**< Through CV or not*/

    WE1AppCfg_GPIO_e motion_led_ind;/**< Motion Led indication*/
    WE1AppCfg_GPIO_e human_detect_ind;/**< human detect Led indication*/
    WE1AppCfg_GPIO_e ir_led;/**< IR LED*/
    WE1AppCfg_Ambient_Light_e light_support;/**< Ambient light Support*/
    WE1AppCfg_PDM_Support_e pdm_support;
    WE1AppCfg_I2S_Support_e i2s_support;
}EMBARC_PACKED WE1AppCfg_APP_t;

typedef struct
{
	SENSORDPLIB_SENSOR_E sensor_id;
	WE1AppCfg_SensorColorType_e sensor_color;
	SENSORDPLIB_STREAM_E sensor_stream_type;
	uint16_t sensor_width;
	uint16_t sensor_height;
	uint16_t active_cfg_cnt;
	HX_CIS_SensorSetting_t sensor_cfg[MAX_SUPPORT_SENSOR_CFG_COUNT];
}EMBARC_PACKED WE1AppCfg_Sensor_t;

typedef struct
{
	uint8_t active_cfg_cnt;
	HX_CIS_SensorSetting_t sensor_stream_cfg[MAX_SENSOR_STREAM_CFG_COUNT];
}EMBARC_PACKED WE1AppCfg_Sensor_StreamOn_t;

typedef struct
{
	uint8_t active_cfg_cnt;
	HX_CIS_SensorSetting_t sensor_off_cfg[MAX_SENSOR_STREAM_CFG_COUNT];
}EMBARC_PACKED WE1AppCfg_Sensor_StreamOff_t;

typedef struct
{
	uint8_t active_cfg_cnt;
	HX_CIS_SensorSetting_t sensor_strobe_on_cfg[MAX_SENSOR_STROBE_CFG_COUNT];
}EMBARC_PACKED WE1AppCfg_Sensor_StrobeOn_t;

typedef struct
{
	uint8_t active_cfg_cnt;
	HX_CIS_SensorSetting_t sensor_strobe_off_cfg[MAX_SENSOR_STROBE_CFG_COUNT];
}EMBARC_PACKED WE1AppCfg_Sensor_StrobeOff_t;

typedef struct
{
	WE1AppCfg_CHIP_Package_e chip_package;
	WE1AppCfg_DP_CLK_Mux_e   dp_clk_mux;
	WE1AppCfg_MCLK_CLK_Mux_e  mclk_clk_mux;
	SENSORCTRL_MCLK_E    mclk_div;
	WE1AppCfg_GPIO_e  xshutdown_pin_sel;
    uint8_t cyclic_buffer_cnt;
    uint32_t wdma1_startaddr;
    uint32_t wdma2_startaddr;
    uint32_t wdma3_startaddr;
    uint32_t jpegsize_autofill_startaddr;
    INP_SUBSAMPLE_E subsample;
	HW2x2_CFG_T hw2x2_cfg;
	CDM_CFG_T cdm_cfg;
	HW5x5_CFG_T hw5x5_cfg;
	JPEG_CFG_T jpeg_cfg;

	uint8_t act_wakupCPU_pin_cnt;
	WE1AppCfg_GPIO_e  wakeupCPU_int_pin[MAX_SUPPORT_WAKEUP_CPU_INT_PIN];
	ADCC_CHANNEL analoge_pir_ch_sel;
	uint32_t analoge_pir_th;
	ADCC_CHANNEL light_sensor_adc_ch_sel;
	uint32_t light_sensor_th;
	PM_CFG_PWR_MODE_E pmu_type;
	PMU_WE1_POWERPLAN_E pmu_powerplan;
	PMU_BOOTROMSPEED_E bootspeed;
	uint8_t 	pmu_skip_bootflow;   /**< Only Support in PM_MODE_CDM_ADC_BOTH, PM_MODE_AOS_ONLY, PM_MODE_ADC_ONLY and ICCM retention on**/
	uint8_t 	support_bootwithcap; /**< Support capture when boot up (PM_MODE_GPIO and PM_MODE_RTC support only)**/
	uint8_t 	s_ext_int_mask;	/**< PMU Sensor External Interrupt Mask **/
}EMBARC_PACKED WE1AppCfg_WE1Driver_t;

typedef struct
{
	WE1AppCfg_SOCCOMType_e comm_type;
}EMBARC_PACKED WE1AppCfg_SOC_COM_t;

typedef struct
{
	WE1AppCfg_TableHeader_t table_header;/**< Table Header */

	WE1AppCfg_TableInfo_t app_table_info;/**< Application Table information */
	WE1AppCfg_TableInfo_t sensor_cfg_table_info;/**< Sensor Configuration Table information */
	WE1AppCfg_TableInfo_t sensor_streamon_table_info;/**< Sensor Stream On Table information */
	WE1AppCfg_TableInfo_t sensor_streamoff_table_info;/**< Sensor Stream Off Table information */
	WE1AppCfg_TableInfo_t we1_table_info;/**< WE-1 Driver Table information */
	WE1AppCfg_TableInfo_t soc_com_table_info;/**< SOC Communication Table information */

	WE1AppCfg_APP_t app_table_cfg;
	WE1AppCfg_Sensor_t  sensor_table_cfg;
	WE1AppCfg_Sensor_StreamOn_t  sensor_streamon_cfg;
	WE1AppCfg_Sensor_StreamOff_t  sensor_streamoff_cfg;
	WE1AppCfg_Sensor_StrobeOn_t  sensor_strobeon_cfg;
	WE1AppCfg_Sensor_StrobeOff_t  sensor_strobeoff_cfg;
	WE1AppCfg_Sensor_t  sensor_md_table_cfg;
	WE1AppCfg_Sensor_StreamOn_t  sensor_md_streamon_cfg;
	WE1AppCfg_WE1Driver_t   we1_driver_cfg;
	WE1AppCfg_SOC_COM_t   soc_comm_cfg;
}EMBARC_PACKED WE1AppCfg_CustTable_t;

#endif /* SCENARIO_APP_IOT_CUSTOM_CONFIG_H_ */
