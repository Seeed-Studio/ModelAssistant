/*
 * tflitemicro_algo.cc
 *
 *  Created on: 20220922
 *      Author: 902453
 */
#include <math.h>
#include <stdint.h>
#include <forward_list>
#include "grove_ai_config.h"
#include "isp.h"
#include "hx_drv_webusb.h"
#include "embARC_debug.h"

#include "tflitemicro_algo.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    std::forward_list<meter_t> _meter_list;

    // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
    // model is preferred over the legacy unsigned model format. This means that
    // throughout this project, input images must be converted from unisgned to
    // signed format. The easiest and quickest way to convert from unsigned to
    // signed 8-bit integers is to subtract 128 from the unsigned value to get a
    // signed value.

    // An area of memory to use for input, output, and intermediate arrays.
    constexpr int kTensorArenaSize = 450 * 1024;
#if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__CCAC__)
    static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
#else
#pragma Bss(".tensor_arena")
    static uint8_t tensor_arena[kTensorArenaSize];
#pragma Bss()
#endif // if defined (_GNUC_) && !defined (_CCAC_)
} // namespace

// TFu debug log, make it hardware platform print
extern "C" void DebugLog(const char *s) { xprintf("%s", s); } //{ fprintf(stderr, "%s", s); }

#define MODEL_INDEX 1
#define ALGORITHM_INDEX 3
#define IMG_PREVIEW_MAX_SIZE 16
#define IMAGE_PREIVEW_ELEMENT_NUM 2
#define IMAGE_PREIVEW_ELEMENT_SIZE 4
#define IMAGE_PREVIEW_FORMATE "{\"type\":\"preview\", \"algorithm\":%d, \"model\":%d,\"count\":%d, \"object\":{\"x\": [%s],\"y\": [%s]}}"

extern "C" int tflitemicro_algo_init()
{
    int ercode = 0;
    uint32_t *xip_flash_addr;

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    xip_flash_addr = (uint32_t *)0x30000000;

    // get model (.tflite) from flash
    model = ::tflite::GetModel(xip_flash_addr);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static tflite::MicroMutableOpResolver<15> micro_op_resolver;
    micro_op_resolver.AddPad();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddConcatenation();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddFullyConnected();

    // Build an interpreter to run the model with.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return -1;
    }
    // Get information about the memory area to use for the model's input.
    input = interpreter->input(0);
    output = interpreter->output(0);

    return ercode;
}

extern "C" int tflitemicro_algo_run(uint32_t img, uint32_t ow, uint32_t oh)
{

    uint16_t h = input->dims->data[1];
    uint16_t w = input->dims->data[2];
    uint16_t c = input->dims->data[3];

    yuv422p2rgb(input->data.uint8, (const uint8_t *)img, oh, ow, c, h, w, VISION_ROTATION);

    for (int i = 0; i < input->bytes; i++)
    {
        input->data.int8[i] -= 128;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
        return -1;
    }

    // Get the results of the inference attempt
    float scale = ((TfLiteAffineQuantization *)(output->quantization.params))->scale->data[0];
    int zero_point = ((TfLiteAffineQuantization *)(output->quantization.params))->zero_point->data[0];

    _meter_list.clear();

    for (int i = 0; i < output->bytes / 2; i++)
    {
        meter_t obj;
        obj.x = uint16_t(float(float(output->data.int8[i * 2] - zero_point) * scale) * ow);
        obj.y = uint16_t(float(float(output->data.int8[i * 2 + 1] - zero_point) * scale) * oh);

        _meter_list.emplace_front(obj);
    }

    return 0;
}

int tflitemicro_algo_get_preview(char *preview, uint16_t max_length)
{
    if (_meter_list.empty())
    {
        return -1;
    }
    uint16_t index = 0;
    // 获取目前结果集长度
    uint16_t size = std::distance(_meter_list.begin(), _meter_list.end());

    // 输入preview最多能有多少element
    uint16_t available_size = (max_length - sizeof(IMAGE_PREVIEW_FORMATE)) / (IMAGE_PREIVEW_ELEMENT_SIZE * IMAGE_PREIVEW_ELEMENT_NUM);

    if (available_size < 1)
    {
        return -1;
    }

    // element数组
    char element[IMAGE_PREIVEW_ELEMENT_NUM][IMG_PREVIEW_MAX_SIZE * IMAGE_PREIVEW_ELEMENT_SIZE] = {0};

    // 生成element
    for (auto it = _meter_list.begin(); it != _meter_list.end(); ++it)
    {
        if (index == 0)
        {
            snprintf(element[0], sizeof(element[0]), "%d", it->x);
            snprintf(element[1], sizeof(element[1]), "%d", it->y);
        }
        else
        {
            snprintf(element[0], sizeof(element[0]), "%s,%d", element[0], it->x);
            snprintf(element[1], sizeof(element[1]), "%s,%d", element[1], it->y);
        }
        index++;
        // 如果超过最大的可预览长度 则退出
        if (index > IMG_PREVIEW_MAX_SIZE || index > available_size)
        {
            break;
        }
    }

    // 规格化preview
    snprintf(preview, max_length, IMAGE_PREVIEW_FORMATE, ALGORITHM_INDEX, MODEL_INDEX, index, element[0], element[1]);

    return 0;
}

void tflitemicro_algo_exit()
{
    hx_lib_pm_cplus_deinit();
}
