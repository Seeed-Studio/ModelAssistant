import serial
import csv

# 打开串行端口
ser = serial.Serial('COM3', 115200)  # 请根据实际情况修改串行端口和波特率
if ser.isOpen():  # 判断串口是否成功打开
    print("打开串口成功。")
    print(ser.name)  # 输出串口号
else:
    print("打开串口失败。")

csv_file = open('serial_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# 打开一个文件用于保存数据
file_path = "serial_data.csv"
with open(file_path, 'a') as file:
    try:
        for index, i in enumerate(ser):
            # 从串行端口读取数据
            if index == 0:
                continue
            data = i.decode().strip('\r\n').split(' ')[:-1]
            # 将数据写入文件
            csv_writer.writerow(data)
            # 打印读取到的数据
            print("Received:", index)
            # str_list = data.split(' ')[:-1]
            # data = np.array(str_list).astype('float32')

    except KeyboardInterrupt:
        # 如果按下 Ctrl+C，则关闭串行端口和文件
        ser.close()
        csv_file.close()
        print("Data saved to:", file_path)
