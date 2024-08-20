import serial

ser = serial.Serial('COM9', 115200)
if ser.isOpen():  # 判断串口是否成功打开
    print("打开串口成功。")
    print(ser.name)  # 输出串口号
else:
    print("打开串口失败。")
while True:
    data = []
    # a = time.time()
    for index, i in enumerate(ser):
        temp = i.decode().strip('\r\n')
        start_index = temp.find("Predict results:")
        end_index = temp.find("Predict time:")
        temp = temp[start_index:end_index]
        if len(temp) > 0:
            temp = temp[16:]
            # print(temp)
            loss1 = (float(temp[0:10]) + 27) * 10
            # print(loss1)
            # print(temp[0:10])
            if float(temp[0:10]) > -30:
                print("正常")
            else:
                print("异常")
