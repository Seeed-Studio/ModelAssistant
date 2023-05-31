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
    def __init__(self, tflite_model_path, log_dir, n_cores=os.cpu_count(), invoke_times=10):
        self.path = tflite_model_path
        self.log_dir = log_dir
        self.n_cores = n_cores
        self.invoke_times = invoke_times

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

        if not self.check_parms():
            return None

        self.logger.info('TFLite model tester runtime informations: \n\tTensorFlow version: %s \n\tTFLite model path: %s \n\tNum threads: %d', tf.__version__, self.path, self.n_cores)
        self.interpreter = tf.lite.Interpreter(model_path=self.path, num_threads=self.n_cores)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.logger.info('TFLite interpreter input/output details: \n\tInput details: %s \n\tOutput details: %s', self.input_details, self.output_details)

    def check_parms(self) -> bool:
        if not f'{self.path}'.endswith('.tflite'):
            self.logger.error('TFLite model path should have a \'.tflite\' file extension, please check the path parameters.')
            return False
        if self.n_cores < 1:
            self.logger.error('Number of threads while invoking should not less than 1, current: %d.', self.n_cores)
            return False
        if self.invoke_times < 1:
            self.logger.error('Times of invoking time test should not less than 1, current: %d.', self.invoke_times)
            return False

        return True

    def invoke(self):
        for inp_detail in self.input_details:
            inp_index = inp_detail['index']
            inp_shape = inp_detail['shape']
            inp_data_type = inp_detail['dtype']
            dummy_input = np.zeros(inp_shape, dtype=inp_data_type)
            self.interpreter.set_tensor(inp_index, dummy_input)

        s = datetime.datetime.now()
        self.interpreter.invoke()
        e = datetime.datetime.now()
        duration = e - s

        output_data = {}
        for out_detail in self.output_details:
            out_index = out_detail['index']
            out_name = out_detail['name']
            output_data[out_name] = self.interpreter.get_tensor(out_index)
        self.interpreter.reset_all_variables()

        return output_data, duration

    def run(self) -> int:
        self.logger.info('Invoking using dummy input: \n\tInputs: %d \n\tOutputs: %d', len(self.input_details), len(self.output_details))
        output_data, _ = self.invoke()

        self.logger.info('Checking if invoke result matches model output: \n\tOutput data: %s', output_data)
        for out_detail in self.output_details:
            out_name = out_detail['name']
            out_shape = out_detail['shape']
            out_dtype = out_detail['dtype']
            if out_name not in output_data.keys():
                self.logger.error('Cannot find output \'%s\' in output data.', out_name)
                return -1
            if (output_data[out_name].shape != out_shape).any():
                self.logger.error('Output \'%s\' shape mismatch: \n\tCurrent: %s \n\tExpected: %s', out_name, output_data[out_name].shape, out_shape)
                return -1
            if output_data[out_name].dtype != out_dtype:
                self.logger.error('Output \'%s\' data type mismatch: \n\tCurrent: %s \n\tExpected: %s', out_name, output_data[out_name].dtype, out_dtype)
                return -1

        model_metadata = self.interpreter.get_tensor_details()
        self.logger.info('Getting TFLite model details: \n\tNodes: %d \n\tContents: %s', len(model_metadata), model_metadata)

        from tensorflow.lite.python.analyzer_wrapper import _pywrap_analyzer_wrapper as _analyzer_wrapper
        model_analytics_result = _analyzer_wrapper.ModelAnalyzer(self.path, True, False)
        self.logger.info('Analyzing TFlite model: \n\tResults: %s', model_analytics_result)

        self.logger.info('Testing TFLite model invoking time: \n\tTimes: %d', self.invoke_times)
        invoke_durations = []
        for _ in range(self.invoke_times):
            _, duration = self.invoke()
            invoke_durations.append(duration.microseconds / 1000)
        self.logger.info('Invoking time statics: \n\tAverage: %.4fms \n\tMin: %.4fms \n\tMax: %.4fms', np.average(invoke_durations), np.min(invoke_durations), np.max(invoke_durations))

        self.logger.info('TFLite model test result: \n\tStatus: Passed \n\tReport path: %s', self.log_path)

        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='EdgeLab TFLite model test script')
    parser.add_argument('model', help='path of the TFLite model file')
    parser.add_argument('--workdir', default='work_dirs', help='path of the work directory')
    parser.add_argument('--num_threads', default='0', type=int, help='number of threads while invoking')
    parser.add_argument('--invoke_times', default='10', type=int, help='times of invoke time test')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    tflite_tester = TFLITE_TESTER(
        args.model,
        args.workdir,
        args.num_threads if args.num_threads > 0 else os.cpu_count(),
        args.invoke_times
    )
    ret = tflite_tester.run()

    return ret

if __name__ == '__main__':
    main()
