import serial
import argparse
import csv
import tools as dt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", "-sr", type=int, default=115200)
    parser.add_argument("--port", "-p", type=str, default="COM5")
    parser.add_argument("--file_path", "-f", type=str, default="serial_data.csv")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    port, sample_rate, file_path = args.port, args.sample_rate, args.file_path
    ser = serial.Serial(port, sample_rate)
    if ser.isOpen():
        print(f"Opening serial port {port} succeeded.")
        print(ser.name)
    else:
        print(f"Opening serial port {port} failed.")

    csv_file = open(file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    with open(file_path, "a") as file:
        try:
            data = []
            record = 0
            if sample_rate is not None:
                for index, i in enumerate(ser):

                    if (index + 1) % sample_rate != 0:
                        temp = i.decode().strip("\r\n").split(" ")[:-1]
                        data = data + temp

                        continue
                    record = record + 1
                    temp = i.decode().strip("\r\n").split(" ")[:-1]
                    data = data + temp

                    csv_writer.writerow(data)
                    print("Received:", record)
                    data = []

            else:
                # ser.reset_input_buffer()
                ser.flushInput()
                for index, i in enumerate(ser):
                    if index == 0:
                        continue
                    temp = i.decode().strip("\r\n").split(" ")[:-1]
                    csv_writer.writerow(temp)
                    print("Received:", index)
                    ser.flushInput()

        except KeyboardInterrupt:
            ser.close()
            csv_file.close()
            print("Data saved to:", file_path)
