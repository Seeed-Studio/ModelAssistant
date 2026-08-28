import os
import os.path as osp
import sys
import shutil
import argparse

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.evaluator import DumpResults

from sscma.utils import lazy_import


def parse_args():
    parser = argparse.ArgumentParser(description="test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument(
        "--img-size",
        "--img_size",
        "--imgsz",
        nargs="+",
        type=int,
        default=[320, 320],
        help="Image size of height and width",
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify onnx model by onnx-sim"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (PyTorch >= 2.9's dynamo exporter supports "
        "opset >= 18 only; lower values trigger a noisy, usually failing "
        "version-conversion fallback)",
    )
    parser.add_argument(
        "--image_path", type=str, help="Used to export verification data of tflite"
    )
    parser.add_argument(
        "--format",
        nargs="*",
        default=["onnx"],
        choices=["onnx", "tflite", "vela", "savemodel", "torchscript", "hailo"],
        help="Model format to be exported",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="hailo8l",
        choices=["hailo8", "hailo8l", "hailo15", "hailo15l"],
        help="hailo hardware type",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify whether the exported tflite results are aligned with the tflitemicro results",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args

def find_and_sample_images(folder_path, limit=10000, sample_size=100):
    import random

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []

    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided image path '{folder_path}' is not a valid directory.")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
                if len(image_files) >= limit:
                    break
        if len(image_files) >= limit:
            break

    found = len(image_files)
    if found < sample_size:
        if found == 0:
            raise ValueError(f"No images found in the directory '{folder_path}'.")
        print(f"Warning: Found only {found} images, which is less than the requested sample size of {sample_size}.")
        sample_size = found

    random_sample = random.sample(image_files, sample_size)

    return random_sample

def generate_input(images_path, img_shape):
    import cv2

    res = []
    datasets = find_and_sample_images(images_path, limit=10000, sample_size=100)
    for ps in datasets[:100]:
        img = cv2.imread(ps)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = cv2.resize(img, (img_shape[0], img_shape[1]))
        # keep a consistent float32 dtype - a mixed float64/float32 array is
        # upcast to float64 and confuses onnx2tf's quantization calibration
        res.append(img.astype(np.float32))
    return np.asarray(res)


def main():
    args = parse_args()
    # verify args
    if (
        True in [fm in ["hailo", "tflite", "vela"] for fm in args.format]
        and args.image_path is None
    ):
        raise ValueError("image_path is required for hailo/tflite/vela format")
    # load config
    cfg = Config.fromfile(args.config, modified_constant=args.cfg_options)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    cfg.load_from = args.checkpoint

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
        shutil.rmtree(runner.log_dir, ignore_errors=True)
        runner._log_dir = osp.dirname(args.checkpoint)

    # else:
    #     # build customized runner from the registry
    #     # if 'runner_type' is set in the cfg
    #     runner = RUNNERS.build(cfg)

    runner.call_hook("before_run")
    runner.model.to(device=args.device)
    runner.load_checkpoint(args.checkpoint, map_location=torch.device(args.device))
    # the runner builds the model in training mode; exporting without eval()
    # would bake batch-statistics BatchNorm into the graph
    model = runner.model.to(device=args.device).eval()

    model_format = args.format
    new_model_format = []
    for fmt in model_format:
        if fmt == "onnx":
            new_model_format.append("onnx")
        elif fmt == "hailo":
            new_model_format.extend(["onnx", "hailo"])
        elif fmt == "savemodel":
            new_model_format.extend(["onnx", "savemodel"])
        elif fmt == "tflite":
            new_model_format.extend(["onnx", "tflite"])
        elif fmt == "vela":
            new_model_format.extend(["onnx", "tflite", "vela"])
        elif fmt == "torchscript":
            new_model_format.append("torchscript")
    new_model_format = list(set(new_model_format))

    calibration_data = None
    if args.image_path:
        if not osp.exists("calibration_image_sample_data_20x128x128x3_float32.npy"):
            input_data = generate_input(args.image_path, args.img_size)
            np.save(
                "calibration_image_sample_data_20x128x128x3_float32.npy", input_data
            )
        # generate_input() produces samples already normalized to [0, 1], so
        # the calibration normalization is a no-op: mean=0, std=1. (Deriving
        # these from the config's data_preprocessor is wrong here - those
        # values describe the model's internal normalization of 0-255 inputs,
        # and a missing std would produce a division by zero.)
        calibration_data = [
            [
                "images",
                "calibration_image_sample_data_20x128x128x3_float32.npy",
                [[[0.0]]],
                [[[1.0]]],
            ]
        ]
    # export
    if "torchscript" in new_model_format:
        export_torchscript(model, args)

    if "onnx" in new_model_format:
        onnx_file = export_onnx(model, args)

    if "hailo" in new_model_format:
        export_hailo(onnx_file, args.arch, args.img_size, cfg, args.image_path)

    if "savemodel" in new_model_format:
        export_savemodel(onnx_file, calibration_data)

    if "tflite" in new_model_format:
        tflite_file = export_tflite(onnx_file, calibration_data)

    if "vela" in new_model_format:
        export_vela(tflite_file, args.verify)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))


@lazy_import("onnx2tf", install_only=True, version=">=2.5.0,<=2.6.8")
@lazy_import("tf-keras", install_only=True)
@lazy_import("onnx-graphsurgeon", install_only=True)
@lazy_import("sng4onnx", install_only=True)
@lazy_import("onnxsim", install_only=True)
def export_savemodel(onnx_file, calibration_data=None):
    # onnx convert to pb
    from onnx2tf import onnx2tf

    try:
        onnx2tf.convert(
            onnx_file,
            output_folder_path=osp.dirname(onnx_file),
            # fix the batch dim: onnx2tf keeps it dynamic by default and then
            # emits RESHAPE ops with computed (non-constant) shapes, which
            # vela rejects ("Does not have valid TFLite Semantics")
            batch_size=1,
            custom_input_op_name_np_data_path=calibration_data,
            output_signaturedefs=True,
            # onnx2tf >= 2.0 defaults to the flatbuffer-direct path and no
            # longer writes a SavedModel unless explicitly requested
            flatbuffer_direct_output_saved_model=True,
            verbosity="warn",
        )
        print("The pb model format was exported successfully")
    except Exception as e:
        print("Export of pb model failed, export interrupted")
        raise RuntimeError(e)

    return osp.dirname(onnx_file)


def export_torchscript(model, args):
    from torch.utils.mobile_optimizer import optimize_for_mobile

    f = f"{osp.splitext(args.checkpoint)[0]}_script.pt"
    script_model = torch.jit.trace(
        model, torch.randn(1, 3, *args.img_size).to(args.device)
    )
    script_model = optimize_for_mobile(script_model)
    torch.jit.save(script_model, f)


@lazy_import("onnx")
@lazy_import("onnxsim", install_only=True)
def export_onnx(model, args):
    import onnx

    fake_input = torch.randn(1, 3, *args.img_size).to(args.device)
    f = f"{osp.splitext(args.checkpoint)[0]}.onnx"
    torch.onnx.export(
        model,
        fake_input,
        f,
        verbose=False,
        input_names=["images"],
        opset_version=args.opset,
    )
    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.simplify:
        try:
            import onnxsim

            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
            onnx.save(onnx_model, f)
        except Exception as e:
            print(f"Simplify failure: {e}")
            raise RuntimeError(e)

    return f


def export_hailo(onnx_path: str, arch: str, img_shape, cfg, img_path):
    from hailo_sdk_client import ClientRunner
    import onnx
    import cv2
    from hailo_sdk_client.exposed_definitions import CalibrationDataType

    datasets = [
        osp.join(img_path, i) for i in os.listdir(img_path) if i.endswith(".jpg")
    ]
    calib_dataset = []
    for ps in datasets[:300]:
        img = cv2.imread(ps)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = cv2.resize(img, (img_shape[0], img_shape[1]))
        calib_dataset.append(img)
    calib_dataset = np.asarray(calib_dataset)

    har_file = f"{osp.dirname(onnx_path)}{osp.sep}{osp.splitext(osp.basename(onnx_path))[0]}.har"
    har_quant_file = f"{osp.dirname(onnx_path)}{osp.sep}{osp.splitext(osp.basename(onnx_path))[0]}_quant.har"
    hef_file = f"{osp.dirname(onnx_path)}{osp.sep}{osp.splitext(osp.basename(onnx_path))[0]}.hef"

    model = onnx.load(onnx_path)
    runner = ClientRunner(hw_arch=arch)
    input_shape = {
        inp.name: [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        for inp in model.graph.input
    }
    runner.translate_onnx_model(
        onnx_path,
        "onnx",
        start_node_names=[i.name for i in model.graph.input],
        end_node_names=[i.name for i in model.graph.output],
        net_input_shapes=input_shape,
    )
    runner.save_har(har_file)
    runner = ClientRunner(har=har_file, hw_arch="hailo8l")

    alls = f"normalization1 = normalization({cfg.model.data_preprocessor.mean}, {cfg.model.data_preprocessor.std})\n"
    runner.load_model_script(alls)
    runner.optimize(calib_dataset, CalibrationDataType.np_array)
    runner.save_har(har_quant_file)
    runner = ClientRunner(har=har_quant_file, hw_arch="hailo8l")
    hef = runner.compile()
    with open(hef_file, "wb") as f:
        f.write(hef)


def optimize_tflite_for_vela(tflite_path: str) -> None:
    """Post-process a quantized TFLite model so the whole graph is legal
    static TFLite that the Vela compiler (and XNNPACK) fully accepts.

    onnx2tf's per-tensor quantizer leaves three classes of problems behind
    (all visible in the YOLOv5 decode tail):

    1. ``shape_signature`` keeps dynamic (-1) dims even though every tensor
       has a static shape. Vela's regor reads the *signature*, reports
       "Dynamic non-batch dimension" and refuses to place any downstream op
       on the NPU. The signatures are synced to the static shapes.
    2. Quantization scales/zero-points drift inside bit-exact op chains
       (RESHAPE/SQUEEZE/EXPAND_DIMS/TRANSPOSE/SLICE/CONCATENATION), which
       both Vela ("Does not have valid TFLite Semantics") and XNNPACK
       reject. Tensors linked by such ops are grouped (union-find) and
       share one parameter set (graph output wins, else the chain source),
       matching the reference TFLite quantizer's same-scale propagation.
    3. Elementwise ops whose constant operand is broadcast along a
       non-batch axis (e.g. the YOLOv5 grid add) are not NPU-placeable;
       the constant is materialized (tiled) to the full shape, and groups
       of constant SLICEs that partition one axis are merged into a single
       SPLIT_V (the form regor handles best).
    """
    import flatbuffers
    import numpy as np
    from onnx2tf.tflite_builder.schema import schema_generated as sgt

    passthrough = {
        sgt.BuiltinOperator.RESHAPE,
        sgt.BuiltinOperator.SQUEEZE,
        sgt.BuiltinOperator.EXPAND_DIMS,
        sgt.BuiltinOperator.TRANSPOSE,
        sgt.BuiltinOperator.SLICE,
    }
    elementwise = {
        sgt.BuiltinOperator.ADD,
        sgt.BuiltinOperator.SUB,
        sgt.BuiltinOperator.MUL,
    }

    with open(tflite_path, 'rb') as f:
        buf = f.read()
    model_t = sgt.ModelT.InitFromObj(sgt.Model.GetRootAsModel(buf, 0))
    n_splitv = n_bcast = n_quant = n_sig = 0

    for subgraph in model_t.subgraphs:
        # --- pass 1: merge constant SLICE partitions into SPLIT_V ---------
        def read_const_i32(t):
            b = model_t.buffers[t.buffer].data
            return np.frombuffer(bytes(b), dtype=np.int32) if b is not None else None

        def add_const_i32(values):
            arr = np.array(values, dtype=np.int32)
            buf_t = sgt.BufferT()
            buf_t.data = arr.tobytes()
            model_t.buffers.append(buf_t)
            t = sgt.TensorT()
            t.type = sgt.TensorType.INT32
            t.shape = np.array(list(arr.shape), dtype=np.int32)
            t.buffer = len(model_t.buffers) - 1
            t.name = f'vela_const_{len(model_t.buffers) - 1}'.encode()
            subgraph.tensors.append(t)
            return len(subgraph.tensors) - 1

        groups = {}
        for idx, op in enumerate(subgraph.operators):
            if model_t.operatorCodes[op.opcodeIndex].builtinCode == sgt.BuiltinOperator.SLICE:
                groups.setdefault(op.inputs[0], []).append(idx)

        splitv_opcode = next(
            (i for i, o in enumerate(model_t.operatorCodes) if o.builtinCode == sgt.BuiltinOperator.SPLIT_V), None
        )
        if splitv_opcode is None:
            oc = sgt.OperatorCodeT()
            oc.builtinCode = sgt.BuiltinOperator.SPLIT_V
            model_t.operatorCodes.append(oc)
            splitv_opcode = len(model_t.operatorCodes) - 1

        to_delete = set()
        for input_t, idxs in groups.items():
            if len(idxs) < 2:
                continue
            ops = [subgraph.operators[i] for i in idxs]
            begins = [read_const_i32(subgraph.tensors[op.inputs[1]]) for op in ops]
            sizes = [read_const_i32(subgraph.tensors[op.inputs[2]]) for op in ops]
            if any(b is None or s is None for b, s in zip(begins, sizes)):
                continue
            axis_set = {int(np.nonzero(s != -1)[0][0]) for s in sizes}
            if len(axis_set) != 1:
                continue
            axis = axis_set.pop()
            in_shape = [int(d) for d in subgraph.tensors[input_t].shape]
            order = sorted(range(len(ops)), key=lambda k: int(begins[k][axis]))
            lens, outs, ok, pos = [], [], True, 0
            for k in order:
                b, ln = int(begins[k][axis]), int(sizes[k][axis])
                if b != pos:
                    ok = False
                    break
                pos += ln
                lens.append(ln)
                outs.append(ops[k].outputs[0])
            if not ok or pos != in_shape[axis]:
                continue
            new_op = sgt.OperatorT()
            new_op.opcodeIndex = splitv_opcode
            new_op.inputs = np.array([input_t, add_const_i32(lens), add_const_i32([axis])], dtype=np.int32)
            new_op.outputs = np.array(outs, dtype=np.int32)
            opts = sgt.SplitVOptionsT()
            opts.numSplits = len(ops)
            new_op.builtinOptionsType = sgt.BuiltinOptions.SplitVOptions
            new_op.builtinOptions = opts
            subgraph.operators[idxs[0]] = new_op
            for i in idxs[1:]:
                to_delete.add(i)
            n_splitv += 1
        if to_delete:
            subgraph.operators = [op for i, op in enumerate(subgraph.operators) if i not in to_delete]

        # --- pass 2: materialize broadcast constants ----------------------
        consumers = {}
        for op in subgraph.operators:
            if op.inputs is not None:
                for i in op.inputs:
                    consumers[i] = consumers.get(i, 0) + 1
        for op in subgraph.operators:
            code = model_t.operatorCodes[op.opcodeIndex].builtinCode
            if code not in elementwise or op.inputs is None or len(op.inputs) != 2:
                continue
            a_t, b_t = subgraph.tensors[op.inputs[0]], subgraph.tensors[op.inputs[1]]
            for const_t, other_t in ((b_t, a_t), (a_t, b_t)):
                buf = model_t.buffers[const_t.buffer]
                const_idx = op.inputs[0] if const_t is a_t else op.inputs[1]
                if buf.data is None or consumers.get(const_idx, 0) != 1:
                    continue
                if const_t.type not in (sgt.TensorType.INT8, sgt.TensorType.UINT8):
                    continue
                c_shape = [int(d) for d in const_t.shape]
                o_shape = [int(d) for d in other_t.shape]
                if c_shape == o_shape or len(c_shape) != len(o_shape):
                    continue
                if all(c == 1 or c == o for c, o in zip(c_shape, o_shape)):
                    data = np.frombuffer(bytes(buf.data), dtype=np.int8).reshape(c_shape)
                    tiled = np.broadcast_to(data, o_shape).copy()
                    new_buf = sgt.BufferT()
                    new_buf.data = tiled.tobytes()
                    model_t.buffers[const_t.buffer] = new_buf
                    const_t.shape = np.array(o_shape, dtype=np.int32)
                    n_bcast += 1
                    break

        # --- pass 3: quant consistency across bit-exact chains ------------
        n_tensors = len(subgraph.tensors)
        parent = list(range(n_tensors))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        producer = {}
        for op in subgraph.operators:
            if op.outputs is not None:
                for o in op.outputs:
                    producer[o] = op

        for op in subgraph.operators:
            if op.inputs is None or op.outputs is None or len(op.inputs) == 0 or len(op.outputs) == 0:
                continue
            opcode = model_t.operatorCodes[op.opcodeIndex].builtinCode
            if opcode in passthrough:
                union(op.inputs[0], op.outputs[0])
            elif opcode == sgt.BuiltinOperator.CONCATENATION:
                for i in op.inputs:
                    union(i, op.outputs[0])
            elif opcode == sgt.BuiltinOperator.SPLIT_V:
                for o in op.outputs:
                    union(op.inputs[0], o)

        uf_groups = {}
        for t in range(n_tensors):
            uf_groups.setdefault(find(t), []).append(t)

        graph_outputs = set(subgraph.outputs) if subgraph.outputs is not None else set()
        for members in uf_groups.values():
            if len(members) < 2:
                continue
            canonical = None
            for t in members:
                if t in graph_outputs and subgraph.tensors[t].quantization is not None:
                    canonical = subgraph.tensors[t].quantization
                    break
            if canonical is None:
                for t in members:
                    prod_op = producer.get(t)
                    prod_code = (
                        model_t.operatorCodes[prod_op.opcodeIndex].builtinCode if prod_op is not None else None
                    )
                    if prod_code not in passthrough and prod_code not in (
                        sgt.BuiltinOperator.CONCATENATION,
                        sgt.BuiltinOperator.SPLIT_V,
                    ):
                        if subgraph.tensors[t].quantization is not None:
                            canonical = subgraph.tensors[t].quantization
                            break
            if canonical is None:
                continue
            for t in members:
                q = subgraph.tensors[t].quantization
                if q is None:
                    continue
                if q.scale != canonical.scale or q.zeroPoint != canonical.zeroPoint:
                    subgraph.tensors[t].quantization = canonical
                    n_quant += 1

        # --- pass 4: sync shape signatures to static shapes ----------------
        for t in subgraph.tensors:
            shape = [int(d) for d in t.shape] if t.shape is not None else []
            if not shape or any(d <= 0 for d in shape):
                continue
            sig = [int(d) for d in t.shapeSignature] if t.shapeSignature is not None else []
            if sig != shape:
                t.shapeSignature = np.array(shape, dtype=np.int32)
                n_sig += 1

    builder = flatbuffers.Builder(0)
    builder.Finish(model_t.Pack(builder), b'TFL3')
    with open(tflite_path, 'wb') as f:
        f.write(builder.Output())
    print(
        f'vela optimization of {osp.basename(tflite_path)}: '
        f'{n_splitv} SLICE->SPLIT_V merge(s), {n_bcast} broadcast constant(s) materialized, '
        f'{n_quant} quant parameter(s) aligned, {n_sig} shape signature(s) synced'
    )


@lazy_import("onnx2tf", install_only=True, version=">=2.5.0,<=2.6.8")
def export_tflite(onnx_path: str, calibration_data=None):
    # Convert ONNX directly to a full-integer-quantized INT8 TFLite model via
    # onnx2tf's flatbuffer-direct path. The previous pipeline (onnx2tf ->
    # SavedModel -> tf.lite.TFLiteConverter.from_saved_model) no longer works:
    # onnx2tf >= 2.0 does not emit SavedModel by default, and its SavedModel
    # exporter is broken with TensorFlow >= 2.20 (depthwise_conv2d dilations).
    # Per-tensor quantization is used: it is what the Ethos-U55 Vela compiler
    # expects, and onnx2tf's strict per-channel validation fails for some ops.
    from onnx2tf import onnx2tf

    # Constant-fold the graph first: computed Reshape shapes (e.g. from the
    # YOLOv5 head's view calls) otherwise survive into the TFLite model as
    # non-constant shape tensors, which Vela rejects.
    try:
        import onnx
        import onnxsim

        onnx_model, ok = onnxsim.simplify(onnx.load(onnx_path))
        if ok:
            onnx.save(onnx_model, onnx_path)
    except Exception as exc:
        print(f'Warning: onnxsim simplification skipped ({exc})')

    file_stem = osp.splitext(osp.basename(onnx_path))[0]
    out_dir = osp.dirname(onnx_path)
    tflite_path = osp.join(out_dir, f"{file_stem}_int8.tflite")

    quantized_file = osp.join(out_dir, f"{file_stem}_full_integer_quant.tflite")
    try:
        onnx2tf.convert(
            onnx_path,
            output_folder_path=out_dir,
            # see export_savemodel: a fixed batch dim keeps RESHAPE shapes
            # constant so vela accepts them
            batch_size=1,
            output_signaturedefs=True,
            verbosity="warn",
            output_integer_quantized_tflite=True,
            quant_type="per-tensor",
            custom_input_op_name_np_data_path=calibration_data,
            input_quant_dtype="int8",
            output_quant_dtype="int8",
        )
    except Exception as exc:
        # onnx2tf also emits an experimental int16-activations variant whose
        # strict validation is broken for depthwise convs (int32 vs int64
        # bias). That failure is raised *after* the int8 artifacts we need
        # have already been written and validated - tolerate it, but only if
        # the target file actually exists.
        if not osp.exists(quantized_file):
            raise
        print(f"Warning: onnx2tf reported an error after writing the INT8 model (ignored): {exc}")

    if not osp.exists(quantized_file):
        raise RuntimeError(
            f"INT8 quantization failed: {quantized_file} was not produced. "
            "Check the onnx2tf log above for details."
        )
    shutil.move(quantized_file, tflite_path)
    optimize_tflite_for_vela(tflite_path)
    print(f"tflite model export successful: {tflite_path}")

    return tflite_path


@lazy_import("ethos-u-vela", install_only=True)
def export_vela(tflite_path: str, verify=False):
    # tflite convert to vela.tflite
    cmd = f"vela \
    --config {osp.dirname(osp.abspath(__file__))}/vela_config.ini \
    --accelerator-config ethos-u55-64 \
    --verbose-performance \
    --system-config My_Sys_Cfg \
    --memory-mode My_Mem_Mode_Parent \
    --output-dir {osp.dirname(tflite_path)}/ \
    {tflite_path}"
    state = os.system(cmd)
    if not state:
        print("Export of vela model succeeded")
    else:
        raise RuntimeError("Export of vela model failed")

    if verify:
        verify_tflite(tflite_path)


def verify_tflite(tflite_path):
    if sys.version_info.major > 2 and sys.version_info.minor > 9:
        import math
        import numpy as np
        import tensorflow as tf
        import tflite_micro as tflm

        sys.setrecursionlimit(1000000)

        tfl_interpreter = tf.lite.Interpreter(
            model_path=tflite_path, experimental_preserve_all_tensors=True
        )
        tfl_interpreter.allocate_tensors()
        input_image = np.random.randint(
            0, 255, tfl_interpreter.get_input_details()[0]["shape_signature"]
        )

        tflm_interpreter = tflm.runtime.Interpreter.from_file(
            tflite_path,
            intrepreter_config=tflm.runtime.InterpreterConfig.kPreserveAllTensors,
        )

        tfl_interpreter.set_tensor(
            tfl_interpreter.get_input_details()[0]["index"], input_image
        )
        tfl_interpreter.invoke()

        tflm_interpreter.set_input(input_image, 0)
        tflm_interpreter.invoke()
        for i, details in enumerate(tfl_interpreter.get_output_details()):
            tflm_tensor = tflm_interpreter.get_output(i)
            tfl_tensor = tfl_interpreter.get_tensor(details["index"])
            is_match = np.allclose(tfl_tensor, tflm_tensor, atol=1, equal_nan=True)
            Accuracy = np.sum(tfl_tensor == tflm_tensor) / math.prod(tfl_tensor.shape)
            string = f'name:{details["name"]} shape:{details["shape"]} Accuracy:{Accuracy} match:{is_match}'
            print(string)

    else:
        print(
            "Using tflite micro requires your Python version to be 3.10 or above.",
            f" Your Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            " cannot be verified and has been skipped.",
        )


if __name__ == "__main__":
    main()
