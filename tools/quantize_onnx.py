from onnxruntime.quantization import QuantType, QuantFormat, StaticQuantConfig, DynamicQuantConfig, QuantizationMode, \
    quantize_static, quantize_dynamic, CalibrationDataReader
import onnx


class DataRead(CalibrationDataReader):
    def __init__(self, model_path):
        super(DataRead, self).__init__(self, model_path)
        pass


onnx_path = r'D:\github\edgelab\work_dirs\tmp.onnx'
quan_path = 'quant.onnx'

quantize_static(onnx_path, quan_path,)
# results = quantize_dynamic(onnx_path, quan_path, weight_type=QuantType.QInt8)
print(results)
