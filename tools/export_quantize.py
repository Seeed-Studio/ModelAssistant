import os
import argparse
import os.path as osp

from onnxmltools.utils import load_model, save_model
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static, QuantFormat

from utils.quant_read import Quan_Reader


def onnx_quant_static(onnx_path):
    onnx_dir = osp.dirname(onnx_path)
    onnx_name = osp.basename(onnx_path)
    quant_file = osp.join(onnx_dir, onnx_name.replace('.onnx', '_quant_static.onnx'))

    # quantize_static(onnx_path,quant_file,quant_format=QuantFormat.QDQ,optimize_model=False,calibration_data_reader=ResNet50DataReader('./img_e',onnx_path),weight_type=QuantType.QInt8)
    quantize_static(onnx_path, quant_file, quant_format=QuantFormat.QDQ, optimize_model=False,
                    calibration_data_reader=Quan_Reader('./img_e', (112, 112), 'images'), weight_type=QuantType.QInt8)
    print('onnx static sucessfull!\nfile in: {}'.format(quant_file))


def onnx_quant_dynamic(onnx_path):
    onnx_dir = osp.dirname(onnx_path)
    onnx_name = osp.basename(onnx_path)
    quant_file = osp.join(onnx_dir, onnx_name.replace('.onnx', '_quant_dynamic.onnx'))
    quantize_dynamic(onnx_path, quant_file, per_channel=True, weight_type=QuantType.QUInt8)
    print('onnx dynamic sucessfull!\nfile in: {}'.format(quant_file))


def export_ncnn(onnx_path):
    onnx_dir = osp.dirname(onnx_path)
    onnx_name = osp.basename(onnx_path)
    ncnn_param = osp.join(onnx_dir, onnx_name.replace('.onnx', '.param'))
    ncnn_bin = osp.join(onnx_dir, onnx_name.replace('.onnx', '.bin'))
    os.system(f'onnx2ncnn {onnx_path} {ncnn_param} {ncnn_bin}')
    print('ncnn export sucessfull!')


def ncnn_quant(onnx_path, image_dir='./img_e', img_size=[112, 112, 3]):
    onnx_dir = osp.dirname(onnx_path)
    onnx_name = osp.basename(onnx_path)
    ncnn_param = osp.join(onnx_dir, onnx_name.replace('.onnx', '.param'))
    ncnn_bin = osp.join(onnx_dir, onnx_name.replace('.onnx', '.bin'))
    ncnn_param_opt = osp.join(onnx_dir, onnx_name.replace('.onnx', '_opt.param'))
    ncnn_bin_opt = osp.join(onnx_dir, onnx_name.replace('.onnx', '_opt.bin'))
    ncnn_table = osp.join(onnx_dir, onnx_name.replace('.onnx', '.table'))
    ncnn_param_int8 = osp.join(onnx_dir, onnx_name.replace('.onnx', '_int8.param'))
    ncnn_bin_int8 = osp.join(onnx_dir, onnx_name.replace('.onnx', '_int8.bin'))
    os.system(f"ncnnoptimize {ncnn_param} {ncnn_bin} {ncnn_param_opt} {ncnn_bin_opt} 0")  # optimizer model

    os.system(f"find {image_dir} -type f > imagelist.txt")  # gener calibration datasets
    os.system(
        f"ncnn2table {ncnn_param_opt} {ncnn_bin_opt} imagelist.txt {ncnn_table} mean=[104,117,123] norm=[0.017,0.017,0.017] shape={img_size} pixel=BGR thread=8 method=kl")  # gener calibration table

    os.system(
        f"ncnn2int8 {ncnn_param_opt} {ncnn_bin_opt} {ncnn_param_int8} {ncnn_bin_int8} {ncnn_table}")  # quantize model

    print('ncnn quantize sucessfull!')


def onnx_fp16(onnx_path):
    onnx_dir = osp.dirname(onnx_path)
    onnx_name = osp.basename(onnx_path)
    onnx_fp16 = osp.join(onnx_dir, onnx_name.replace('.onnx', '_fp16.onnx'))
    onnx_model = load_model(onnx_path)
    fp16_model = convert_float_to_float16(onnx_model)
    save_model(fp16_model, onnx_fp16)
    print('export onnx fp16 format sucessfull!')


def ncnn_fp16(onnx_path):
    onnx_dir = osp.dirname(onnx_path)
    onnx_name = osp.basename(onnx_path)
    ncnn_param = osp.join(onnx_dir, onnx_name.replace('.onnx', '.param'))
    ncnn_bin = osp.join(onnx_dir, onnx_name.replace('.onnx', '.bin'))
    ncnn_param_opt = osp.join(onnx_dir, onnx_name.replace('.onnx', '_fp16.param'))
    ncnn_bin_opt = osp.join(onnx_dir, onnx_name.replace('.onnx', '_fp16.bin'))

    os.system(f"ncnnoptimize {ncnn_param} {ncnn_bin} {ncnn_param_opt} {ncnn_bin_opt} 65536")
    print('export ncnn fp16 format sucessfull!')


def main(args):
    func_dict = {'onnx_fp16': onnx_fp16, 'onnx_quan_st': onnx_quant_static, 'onnx_quan_dy': onnx_quant_dynamic,
                 'ncnn': export_ncnn, \
                 'ncnn_fp16': ncnn_fp16, 'ncnn_quan': ncnn_quant}
    onnx_path = onnx_path = osp.abspath(args.onnx_path)
    export_type = args.export_type
    imags_dir = args.imags_path
    print(imags_dir)

    assert len(export_type), "At least one export type needs to be selected"

    for f in export_type:
        if f == 'ncnn_quan' and imags_dir:
            func_dict[f](onnx_path, imags_dir)
        else:
            func_dict[f](onnx_path)


def args_parse():
    args = argparse.ArgumentParser(description='export onnx to othaer.')
    args.add_argument('--onnx-path', default='./weights/best.onnx', help='onnx model file path')
    args.add_argument('--imags-path', help='celacrater data file path')
    args.add_argument('--export-type', nargs='+', default=['onnx_fp16', 'ncnn_quan'],
                      help='from [onnx_fp16, onnx_quan_st, onnx_quan_dy, ncnn, ncnn_fp16, ncnn_quan]')

    return args.parse_args()


if '__main__' == __name__:
    args = args_parse()
    main(args=args)
    # onnx_path = onnx_path = osp.abspath(args.onnx_path)
    # onnx_fp16(onnx_path)
    # onnx_quant_dynamic(onnx_path)
    # onnx_quant_static(onnx_path)
    # export_ncnn(onnx_path)
    # ncnn_fp16(onnx_path)
    # ncnn_quant(onnx_path)
