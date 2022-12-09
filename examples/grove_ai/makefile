##
# Application makefile
##


RM = rm -rf
DL = curl -Ls -f
UNZIP = unzip 

SDK_LINK = https://files.seeedstudio.com/grove_ai_vision/grove_ai_sdk_v01.zip
SDK_LOC = .
SDK_NAME = grove_ai_sdk_v01.zip


APP?=meter
HW?=grove_ai
##
# root dir of embARC
##
EMBARC_ROOT = .
COMPONENT_ROOT = components
APP_ROOT ?= ./examples/$(APP)

download:
	@echo 'downloading'
	@$(DL) $(SDK_LINK)  -o $(SDK_NAME)
	@$(UNZIP) -o $(SDK_NAME) -d $(SDK_LOC) 
	@$(RM) $(SDK_NAME)


#user define linker script
USER_LINKER_SCRIPT_FILE = $(COMPONENT_ROOT)/linker/$(HW).ld


include $(COMPONENT_ROOT)/component.mk

include $(APP_ROOT)/app.mk


COMMON_COMPILE_PREREQUISITES += makefile


override EMBARC_ROOT := $(strip $(subst \,/,$(EMBARC_ROOT)))
-include $(EMBARC_ROOT)/himax_we1_sdk/options/options.mk
