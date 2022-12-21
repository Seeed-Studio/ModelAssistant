
# USING Pre-built CV library
# 1 for enable
# other for disable
##
SCENARIO_APP_ROOT = $(APP_ROOT)/
LIB_CV_PREBUILT = 0

LIB_SEL += tflitemicro_25

#if you don't use the OV camera, pls comment this code
ifeq ($(HW), grove_vision_ai)
	APPL_DEFINES += -DCIS_OV_SENSOR -DCIS_OV2640_BAYER
endif

LIB_SEL += tflitemicro_25

SCENARIO_APP_INCDIR = $(APP_ROOT)

SCENARIO_APP_SUPPORT_SORTED = $(sort $(SCENARIO_APP_SUPPORT_LIST))		
SCENARIO_APP_INCDIR += $(SCENARIO_APP_SUPPORT_SORTED)


SCENARIO_APP_CSRCS = $(call get_csrcs, $(SCENARIO_APP_INCDIR))
SCENARIO_APP_CXXSRCS = $(call get_cxxsrcs, $(SCENARIO_APP_INCDIR))
SCENARIO_APP_CCSRCS = $(call get_ccsrcs, $(SCENARIO_APP_INCDIR))
SCENARIO_APP_ASMSRCS = $(call get_asmsrcs, $(SCENARIO_APP_ASMSRCDIR))