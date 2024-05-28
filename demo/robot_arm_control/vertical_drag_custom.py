import customtkinter
import json
import serial

customtkinter.set_appearance_mode("dark")

line_last_x = 0
line_last_y = 0

current_x = 0
current_y = 0

slider_value = 150

is_mouse_pressed = False
moving_ctrl_flag = False

ser = None

serial_status = False

line_color = "#A9BCD0"
line2_color = "#4E5973"
draw_color = "#58A4B0"
background_color = "#373F51"

current_sent_y = 0
current_sent_z = 200


def moveto(x, y):
    global current_sent_z, current_sent_y
    if not moving_ctrl_flag or not serial_status:
        return
    data = {
        "T": 1041,
        "x": slider_value,
        "y": -x,
        "z": y,
        "t": 3.14,
    }
    json_data = json.dumps(data)
    ser.write(json_data.encode() + b'\n')
    current_sent_y = -x
    current_sent_z = y
    print(json_data)

def on_mouse_motion(event):
    global line_last_x, line_last_y
    current_x, current_y = event.x - center_x, center_y - event.y
    moveto(current_x, current_y)
    if is_mouse_pressed:
        draw_motion(current_x, current_y)
    else:
        line_last_x = current_x
        line_last_y = current_y

def on_mouse_press(event):
    global is_mouse_pressed
    json_data = mouse_down_entry.get()
    ser.write(json_data.encode() + b'\n')
    print(json_data)
    is_mouse_pressed = True

def on_mouse_release(event):
    global is_mouse_pressed, line_last_x, line_last_y
    json_data = mouse_up_entry.get()
    ser.write(json_data.encode() + b'\n')
    print(json_data)
    line_last_x = current_x
    line_last_y = current_y
    is_mouse_pressed = False

def draw_motion(x, y):
    global line_last_x, line_last_y
    canvas.create_line(center_x + line_last_x, center_y - line_last_y, center_x + x, center_y - y, fill=draw_color, width=2)
    line_last_x = x
    line_last_y = y

def draw_axis():
    canvas.create_line(500, 0, 500, canvas_height, fill=line2_color, width=1)
    canvas.create_line(600, 0, 600, canvas_height, fill=line2_color, width=1)
    canvas.create_line(700, 0, 700, canvas_height, fill=line2_color, width=1)

    canvas.create_line(100, 0, 100, canvas_height, fill=line2_color, width=1)
    canvas.create_line(200, 0, 200, canvas_height, fill=line2_color, width=1)
    canvas.create_line(300, 0, 300, canvas_height, fill=line2_color, width=1)


    canvas.create_line(0, 100, canvas_width, 100, fill=line2_color, width=1)
    canvas.create_line(0, 200, canvas_width, 200, fill=line2_color, width=1)
    canvas.create_line(0, 300, canvas_width, 300, fill=line2_color, width=1)

    canvas.create_line(0, 500, canvas_width, 500, fill=line2_color, width=1)
    canvas.create_line(0, 600, canvas_width, 600, fill=line2_color, width=1)
    canvas.create_line(0, 700, canvas_width, 700, fill=line2_color, width=1)

    canvas.create_line(center_x, 0, center_x, canvas_height, fill=line_color, width=1)
    canvas.create_line(0, center_y, canvas_width, center_y, fill=line_color, width=1)

    canvas.create_text(center_x + 8, 12, text="Z+", fill=line_color, anchor='w')
    canvas.create_text(8, center_y - 12, text="Y+", fill=line_color, anchor='w')

    canvas.create_text(center_x-4, center_y+8, text="0", fill=line_color, anchor='e')
    canvas.create_text(center_x-4-100, center_y+8, text="100", fill=line_color, anchor='e')
    canvas.create_text(center_x-4-200, center_y+8, text="200", fill=line_color, anchor='e')
    canvas.create_text(center_x-4-300, center_y+8, text="300", fill=line_color, anchor='e')
    canvas.create_text(center_x-4-400+30, center_y+8, text="400", fill=line_color, anchor='e')

    canvas.create_text(center_x-4+100, center_y+8, text="100", fill=line_color, anchor='e')
    canvas.create_text(center_x-4+200, center_y+8, text="200", fill=line_color, anchor='e')
    canvas.create_text(center_x-4+300, center_y+8, text="300", fill=line_color, anchor='e')
    canvas.create_text(center_x-4+400, center_y+8, text="400", fill=line_color, anchor='e')

    canvas.create_text(center_x-4, center_y+8-100, text="100", fill=line_color, anchor='e')
    canvas.create_text(center_x-4, center_y+8-200, text="200", fill=line_color, anchor='e')
    canvas.create_text(center_x-4, center_y+8-300, text="300", fill=line_color, anchor='e')
    canvas.create_text(center_x-4, center_y+8-400, text="400", fill=line_color, anchor='e')

    canvas.create_text(center_x-4, center_y+8+100, text="-100", fill=line_color, anchor='e')
    canvas.create_text(center_x-4, center_y+8+200, text="-200", fill=line_color, anchor='e')
    canvas.create_text(center_x-4, center_y+8+300, text="-300", fill=line_color, anchor='e')
    canvas.create_text(center_x-4, center_y+8+400-17, text="-400", fill=line_color, anchor='e')

def clear_lines():
    canvas.delete("all")
    draw_axis()

def space_clear_lines(event):
    clear_lines()

def moving_ctrl():
    global moving_ctrl_flag
    moving_ctrl_flag = not moving_ctrl_flag
    if moving_ctrl_flag:
        moving_button.configure(text="MovingCtrl: Enable", fg_color="#63A375")
    else:
        moving_button.configure(text="MovingCtrl: Disable", fg_color="#D57A66")

def connect_serial():
    global ser, serial_status
    if not serial_status:
        if ser:
            ser.setRTS(False)
            ser.setDTR(False)
            ser.close()
        selected_port = port_entry.get()
        baud_rate = baud_rate_entry.get()
        try:
            ser = serial.Serial(selected_port, baudrate=int(baud_rate), timeout=0.1, dsrdtr=None)
            ser.setRTS(False)
            ser.setDTR(False)
            print(f"Connected to {selected_port} at {baud_rate} baud")
            serial_status = True
        except Exception as e:
            print(f"Failed to connect to {selected_port}: {str(e)}")
            serial_status = False
            port_entry.configure(state="normal")
            connect_button.configure(text="Connect")
            ser.close()
            port_status_lable.configure(text="Disconnected")
        port_entry.configure(state="disabled")
        connect_button.configure(text="Disconnect")
        port_status_lable.configure(text="Connected")
    else:
        ser.close()
        port_entry.configure(state="normal")
        connect_button.configure(text="Connect")
        serial_status = False
        port_status_lable.configure(text="Disconnected")

def slider_event(value):
    global slider_value
    print(value)
    slider_value = value
    slider_label.configure(text=" Z-AXIS VALUE: %d"%slider_value)
    if not serial_status:
        return
    data = {
        "T": 1041,
        "x": slider_value,
        "y": current_sent_y,
        "z": current_sent_z,
        "t": 3.14,
    }
    json_data = json.dumps(data)
    ser.write(json_data.encode() + b'\n')
    print(json_data)


def on_mouse_scroll(event):
    global slider_value
    print("mouse scrolled")
    delta = event.delta
    if delta > 0:
        print("up")
        slider_value += 10
        slider.set(slider_value)
    elif delta < 0:
        print("down")
        slider_value -= 10
        slider.set(slider_value)
    slider_label.configure(text=" Z-AXIS VALUE: %d"%slider_value)
    if not serial_status:
        return
    data = {
        "T": 1041,
        "x": slider_value,
        "y": current_sent_y,
        "z": current_sent_z,
        "t": 3.14,
    }
    json_data = json.dumps(data)
    ser.write(json_data.encode() + b'\n')
    print(json_data)


def on_enter_key(event):
    print("enter")
    if not serial_status:
        connect_serial()
    else:
        moving_ctrl()


# main windows
root = customtkinter.CTk()
root.title("Vertical Drag")
root.bind("<Return>", on_enter_key)

# create a 800x800 Canvas
canvas_width = 800
canvas_height = 800
center_x = canvas_width // 2
center_y = canvas_height // 2

canvas = customtkinter.CTkCanvas(root, width=canvas_width, height=canvas_height, bg=background_color)
canvas.grid(row=0, column=0, padx=20, pady=20)
draw_axis()

# bind input event
canvas.bind("<Motion>", on_mouse_motion)
canvas.bind("<ButtonPress-1>", on_mouse_press)
canvas.bind("<ButtonRelease-1>", on_mouse_release)
canvas.bind("<MouseWheel>", on_mouse_scroll)
root.bind("<space>", space_clear_lines)

# creat a input panel frame
canvas_frame = customtkinter.CTkFrame(root)
canvas_frame.grid(row=0, column=2, padx=20, pady=10)

port_label = customtkinter.CTkLabel(canvas_frame, text="Input Port:")
port_label.grid(row=0, column=0, padx=10, pady=1, sticky="w")

port_entry = customtkinter.CTkEntry(canvas_frame)
port_entry.grid(row=1, column=0, padx=10, pady=1)

port_status_lable = customtkinter.CTkLabel(canvas_frame, text="Disconnected")
port_status_lable.grid(row=1, column=1, padx=10, pady=1, sticky="w")

baud_rate_label = customtkinter.CTkLabel(canvas_frame, text="Baud Rate:")
baud_rate_label.grid(row=2, column=0, padx=10, pady=1, sticky="w")

baud_rate_entry = customtkinter.CTkEntry(canvas_frame)
baud_rate_entry.grid(row=3, column=0, padx=10, pady=1)
baud_rate_entry.insert(0, "115200")

connect_button = customtkinter.CTkButton(canvas_frame, text="Connect", command=connect_serial, fg_color=None)
connect_button.grid(row=3, column=1, padx=10, pady=0)


clear_button = customtkinter.CTkButton(canvas_frame, text="ClearCanvas", command=clear_lines)
clear_button.grid(row=5, column=0, padx=10, pady=10)

moving_button = customtkinter.CTkButton(canvas_frame, text="MovingCtrl: Disable", command=moving_ctrl, fg_color="#D57A66")
moving_button.grid(row=5, column=1, padx=10, pady=10, sticky="w")


down_label = customtkinter.CTkLabel(canvas_frame, text="SET MOUSE DOWN CMD", fg_color="transparent")
down_label.grid(row=6, column=0, padx=10, pady=0, sticky="w")
mouse_down_entry = customtkinter.CTkEntry(canvas_frame, placeholder_text="{\"T\":114,\"led\":255}")
mouse_down_entry.grid(row=7, column=0, padx=10, pady=0, sticky="ew", columnspan=2)
mouse_down_entry.insert(0, "{\"T\":114,\"led\":255}")

up_label = customtkinter.CTkLabel(canvas_frame, text="SET MOUSE UP CMD", fg_color="transparent")
up_label.grid(row=8, column=0, padx=10, pady=0, sticky="w")
mouse_up_entry = customtkinter.CTkEntry(canvas_frame, placeholder_text="{\"T\":114,\"led\":0}")
mouse_up_entry.grid(row=9, column=0, padx=10, pady=0, sticky="ew", columnspan=2)
mouse_up_entry.insert(0, "{\"T\":114,\"led\":0}")

placeholder_1_label = customtkinter.CTkLabel(canvas_frame, text="", fg_color="transparent")
placeholder_1_label.grid(row=10, column=0, padx=10, pady=0, sticky="w")

slider_label = customtkinter.CTkLabel(canvas_frame, text=" X-AXIS VALUE: %d"%slider_value, fg_color="transparent")
slider_label.grid(row=11, column=0, padx=10, pady=0, sticky="w")

slider = customtkinter.CTkSlider(canvas_frame, from_=-400, to=400, command=slider_event)
slider.grid(row=12, column=0, padx=10, pady=0, sticky="ew", columnspan=2)
slider.set(slider_value)

placeholder_2_label = customtkinter.CTkLabel(canvas_frame, text="", fg_color="transparent")
placeholder_2_label.grid(row=13, column=0, padx=10, pady=0, sticky="w")

placeholder_3_label = customtkinter.CTkLabel(canvas_frame, text="1.Please input the port number (COM) to which the robotic arm is connected to the computer in the 'Input Port' section.", justify="left", wraplength=300, fg_color="transparent")
placeholder_3_label.grid(row=14, column=0, padx=10, pady=10, sticky="w", columnspan=2)

placeholder_4_label = customtkinter.CTkLabel(canvas_frame, text="2.Then, click the blue 'Connect' button or press the Enter key to establish a connection. Once connected, you will see the status change from 'Disconnected' to 'Connected' next to it.", justify="left", wraplength=300, fg_color="transparent")
placeholder_4_label.grid(row=15, column=0, padx=10, pady=10, sticky="w", columnspan=2)

placeholder_5_label = customtkinter.CTkLabel(canvas_frame, text="3.After clicking the 'MovingCtrl:Disable' button, the status will change to 'MovingCtrl:Enable'.At this point, you can control the rotation of the robotic arm by moving the mouse in the drawing area on the left.", wraplength=300, justify="left", fg_color="transparent")
placeholder_5_label.grid(row=16, column=0, padx=10, pady=10, sticky="w", columnspan=2)

placeholder_6_label = customtkinter.CTkLabel(canvas_frame, text="4.Left-click on the mouse in the drawing area to turn on the LED light. Release the left mouse button to turn off the LED light. ", wraplength=300, justify="left", fg_color="transparent")
placeholder_6_label.grid(row=17, column=0, padx=10, pady=10, sticky="w", columnspan=2)

placeholder_7_label = customtkinter.CTkLabel(canvas_frame, text="5.When you click the left mouse button and move it in the drawing area, it will leave a trace on the interface. Press the spacebar or click 'ClearCanvas' to clear all traces.", wraplength=300, justify="left", fg_color="transparent")
placeholder_7_label.grid(row=18, column=0, padx=10, pady=10, sticky="w", columnspan=2)

placeholder_8_label = customtkinter.CTkLabel(canvas_frame, text="6.You can control the vertical movement of the robotic arm by scrolling the mouse wheel.", wraplength=300, justify="left", fg_color="transparent")
placeholder_8_label.grid(row=19, column=0, padx=10, pady=10, sticky="w", columnspan=2)

root.mainloop()
