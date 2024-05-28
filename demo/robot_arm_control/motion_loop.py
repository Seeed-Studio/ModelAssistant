# import customtkinter
import json
import serial
import random
import time


line_last_x = 0
line_last_y = 0

current_x = 0
current_y = 0

slider_value = 50

is_mouse_pressed = False
moving_ctrl_flag = False

ser = None

serial_status = False

current_sent_x = 200
current_sent_y = 0




def moveto(x, y, z):
    global current_sent_x, current_sent_y
    if not serial_status:
        return
    data = {
        "T": 1041,
        "x": x,
        "y": y,
        "z": z,
        "t": 3.14,
    }
    json_data = json.dumps(data)
    ser.write(json_data.encode() + b'\n')
    current_sent_x = x
    current_sent_y = y
    print(json_data)


def main_tancle_loop(base_x, base_y, base_z):
    factor = 0
    wait_sec = 0.8
    expan = 0.5
    moveto(base_x + 0, base_y + 200, base_z + 300)
    time.sleep(1)
    while True:
        noisy_x = random.random()*factor
        noisy_y = random.random()*factor
        moveto(base_x + 0*expan + noisy_x, base_y + 0*expan + noisy_y, base_z + 100*expan)
        time.sleep(wait_sec)
        noisy_x = random.random()*factor
        noisy_y = random.random()*factor
        moveto(base_x + 0*expan + noisy_x , base_y + 200*expan + noisy_y, base_z + 200*expan)
        time.sleep(wait_sec)
        noisy_x = random.random()*factor
        noisy_y = random.random()*factor
        moveto(base_x + 200*expan + noisy_x , base_y + 200*expan + noisy_y, base_z + 300*expan)
        time.sleep(wait_sec)
        noisy_x = random.random()*factor
        noisy_y = random.random()*factor
        moveto(base_x + 200*expan + noisy_x, base_y + 0*expan + noisy_y, base_z + 100*expan)
        time.sleep(wait_sec)

selected_port = "COM6"
baud_rate = 115200

try:
    ser = serial.Serial(selected_port, baudrate=int(baud_rate), timeout=0.1, dsrdtr=None)
    ser.setRTS(False)
    ser.setDTR(False)
    print(f"Connected to {selected_port} at {baud_rate} baud")
    serial_status = True
except Exception as e:
    print(f"Failed to connect to {selected_port}: {str(e)}")
    serial_status = False
    ser.close()

                          
main_tancle_loop(-115, -210, -50)
