# 模型训练

SSCMA 提供了多种算法，您可以根据自己的需求选择合适的算法，然后通过训练、导出和部署模型来解决实际问题。本章将进一步介绍如何使用 SSCMA 来训练、导出和部署模型。


## 训练参数说明

您需要将以下部分参数根据实际情况进行替换，各个不同参数的具体说明如下:

```sh
python3 tools/train.py --help

usage: train.py [-h] [--amp] [--auto-scale-lr] [--resume] [--work_dir WORK_DIR] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
                config

Train a detector

positional arguments:
  config                train config file path

options:
  -h, --help            show this help message and exit
  --amp                 enable automatic-mixed-precision training
  --auto-scale-lr       enable automatically scaling LR.
  --resume              resume from the latest checkpoint in the work_dir automatically
  --work_dir WORK_DIR, --work-dir WORK_DIR
                        the dir to save logs and models
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into
                        config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also
                        allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and
                        that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```


## 导出参数说明

您需要将以下部分参数根据实际情况进行替换，各个不同参数的具体说明如下:

```sh
python3 tools/export.py --help

usage: export.py [-h] [--work-dir WORK_DIR] [--out OUT] [--device DEVICE] [--img-size IMG_SIZE [IMG_SIZE ...]] [--simplify] [--opset OPSET]
                 [--image_path IMAGE_PATH] [--format [{onnx,tflite,vela,savemodel,torchscript,hailo} ...]]
                 [--arch {hailo8,hailo8l,hailo15,hailo15l}] [--verify] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                 [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
                 config checkpoint

test (and eval) a model

positional arguments:
  config                test config file path
  checkpoint            checkpoint file

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   the directory to save the file containing evaluation metrics
  --out OUT             dump predictions to a pickle file for offline evaluation
  --device DEVICE       dump predictions to a pickle file for offline evaluation
  --img-size IMG_SIZE [IMG_SIZE ...], --img_size IMG_SIZE [IMG_SIZE ...], --imgsz IMG_SIZE [IMG_SIZE ...]
                        Image size of height and width
  --simplify            Simplify onnx model by onnx-sim
  --opset OPSET         ONNX opset version
  --image_path IMAGE_PATH
                        Used to export verification data of tflite
  --format [{onnx,tflite,vela,savemodel,torchscript,hailo} ...]
                        Model format to be exported
  --arch {hailo8,hailo8l,hailo15,hailo15l}
                        hailo hardware type
  --verify              Verify whether the exported tflite results are aligned with the tflitemicro results
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If
                        the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple
                        values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK

```


## PTQ 参数说明

您需要将以下部分参数根据实际情况进行替换，各个不同参数的具体说明如下:

```sh
python3 tools/quantization.py --help

usage: quantization.py [-h] [--work-dir WORK_DIR] [--test] [--out OUT] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--launcher {none,pytorch,slurm,mpi}]
                       [--local_rank LOCAL_RANK]
                       config model

quantizer a model

positional arguments:
  config                quantization config file path
  model                 checkpoint file

options:
  -h, --help            show this help message and exit
  --work-dir WORK_DIR   the directory to save the file containing evaluation metrics
  --test                Whether to evaluate inference results
  --out OUT             dump predictions to a pickle file for offline evaluation
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a
                        list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are
                        necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```
