import serial
import csv
import tools as dt

# 打开串行端口
ser = serial.Serial('COM8', 115200)  # 请根据实际情况修改串行端口和波特率
if ser.isOpen():  # 判断串口是否成功打开
    print("打开串口成功。")
    print(ser.name)  # 输出串口号
else:
    print("打开串口失败。")

file_path = "serial_data.csv"
csv_file = open(file_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
with open(file_path, 'a') as file:
    try:
        data = []
        record = 0
        for index, i in enumerate(ser):
            # 从串行端口读取数据
            if (index + 1) % dt.sample_rate != 0:
                temp = i.decode().strip('\r\n').split(' ')
                data = data + temp

                continue
            record = record + 1
            temp = i.decode().strip('\r\n').split(' ')
            data = data + temp
            # 将数据写入文件
            csv_writer.writerow(data)
            # 打印读取到的数据
            print("Received:", record)
            data = []
            # str_list = data.split(' ')[:-1]
            # data = np.array(str_list).astype('float32')

    except KeyboardInterrupt:
        # 如果按下 Ctrl+C，则关闭串行端口和文件
        ser.close()
        csv_file.close()
        print("Data saved to:", file_path)
