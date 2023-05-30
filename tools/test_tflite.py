import argparse
import datetime
import os
import logging

import tensorflow as tf
import numpy as np


class LOGGER(logging.Handler):
    def emit(self, record):
        print(self.format(record))


class TFLITE_TESTER:
    def __init__(self, tflite_model_path, log_dir):
        self.path = tflite_model_path
        self.log_dir = log_dir

        folder_name = os.path.dirname(os.path.curdir)
        model_name = os.path.basename(tflite_model_path)
        date_string = datetime.datetime.now().strftime('%Y-%m-%d-%s')
        log_file_name = f'{model_name}_{date_string}.log'
        log_path = os.path.join(folder_name, log_dir, log_file_name)

        if not os.path.exists(os.path.join(folder_name, log_dir)):
            os.makedirs(os.path.join(folder_name, log_dir))

        logging.basicConfig(
            level=logging.DEBUG,
            filename=log_path,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.log_path = log_path
        self.logger = logging.getLogger()
        self.logger.addHandler(LOGGER())

    def run(self):
        logging.info(f'Loading TFLite model from path: {self.path}.')
        interpreter = tf.lite.Interpreter(model_path=self.path)
        interpreter.allocate_tensors()

        logging.info(f'Getting input/output details from TFLite interpreter.')
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.info(f'Input details: \n\t{input_details}')
        logging.info(f'Output details: \n\t{output_details}')

        # TODO: support multiple input/output
        logging.info(f'Invoke testing using dummy input.')
        input_shape = input_details[0]['shape']
        input_data_type = input_details[0]['dtype']
        dummy_input = np.zeros(input_shape, dtype=input_data_type)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        logging.info(f'Checking if invoke result matches model output.')
        output_shape = output_details[0]['shape']
        output_data_type = output_details[0]['dtype']
        if (output_data.shape != output_shape).any():
            logging.error(f'Output shape mismatch: {output_data.shape} != {output_shape}')
        if output_data.dtype != output_data_type:
            logging.error(f'Output data type mismatch: {output_data.dtype} != {output_data_type}')
        logging.info(f'Output shape and data type: \n\t{output_shape}, {output_data_type}')

        logging.info(f'Calculating model parameters.')
        model_metadata = interpreter.get_tensor_details()
        parameters = {
            'size': 0,
            'bytes': 0
        }
        for metadata in model_metadata:
            tensor_dtype = metadata['dtype']
            tensor_bytes = np.dtype(tensor_dtype).itemsize
            layer_shape = metadata['shape']
            layer_size = np.prod(layer_shape)
            parameters['size'] += layer_size
            parameters['bytes'] += tensor_bytes * layer_size
        logging.info(f'Parameters: \n\t{parameters}')

        logging.info(f'Test result stored at: \n\t{self.log_path}')
        logging.info(f'Test passed.')


def parse_args():
    parser = argparse.ArgumentParser(description='EdgeLab TFLite model test script')
    parser.add_argument('model', help='path of the TFLite model file')
    parser.add_argument('--workdir', default='work_dirs', help='path of the work directory')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    tflite_tester = TFLITE_TESTER(args.model, args.workdir)
    tflite_tester.run()


if __name__ == '__main__':
    main()
