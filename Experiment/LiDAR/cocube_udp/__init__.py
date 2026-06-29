import socket
import time
import math
from threading import Thread
import uuid
import numpy as np
from collections import deque


PACKET_SIZE = 47
GRID_TO_MM = 1.35
MAX_DISTANCE = 400
OUTLIER_THRESHOLD = 600

CrcTable = [
    0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae, 0xf2, 0xbf, 0x68, 0x25,
    0x8b, 0xc6, 0x11, 0x5c, 0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07,
    0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5, 0x1f, 0x52, 0x85, 0xc8,
    0x66, 0x2b, 0xfc, 0xb1, 0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
    0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18, 0x44, 0x09, 0xde, 0x93,
    0x3d, 0x70, 0xa7, 0xea, 0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90,
    0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62, 0x97, 0xda, 0x0d, 0x40,
    0xee, 0xa3, 0x74, 0x39, 0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
    0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f, 0xd3, 0x9e, 0x49, 0x04,
    0xaa, 0xe7, 0x30, 0x7d, 0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26,
    0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4, 0x7c, 0x31, 0xe6, 0xab,
    0x05, 0x48, 0x9f, 0xd2, 0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
    0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b, 0x27, 0x6a, 0xbd, 0xf0,
    0x5e, 0x13, 0xc4, 0x89, 0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd,
    0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f, 0xca, 0x87, 0x50, 0x1d,
    0xb3, 0xfe, 0x29, 0x64, 0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
    0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec, 0xb0, 0xfd, 0x2a, 0x67,
    0xc9, 0x84, 0x53, 0x1e, 0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45,
    0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7, 0x5d, 0x10, 0xc7, 0x8a,
    0x24, 0x69, 0xbe, 0xf3, 0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
    0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a, 0x06, 0x4b, 0x9c, 0xd1,
    0x7f, 0x32, 0xe5, 0xa8
]

def CalCRC8(data):
    crc = 0
    for byte in data:
        crc = CrcTable[(crc ^ byte) & 0xFF]
    return crc

def parse_ld06(data):

    if data[0] != 0x54:
        return None
    crc_calc = CalCRC8(data[:PACKET_SIZE-1]) 
    crc_recv = data[PACKET_SIZE-1]
    if crc_calc != crc_recv:
        print(f"CRC check failed: calculated {crc_calc}, received {crc_recv}")
        return None

    start_angle = int.from_bytes(data[4:6], byteorder='little') / 100.0
    end_angle = int.from_bytes(data[42:44], byteorder='little') / 100.0
    if end_angle < start_angle:
        end_angle += 360
    delta_angle = end_angle - start_angle
    points = []
    step = delta_angle / 11.0
    
    for i in range(12):
        offset = 6 + i * 3
        distance = int.from_bytes(data[offset:offset+2], byteorder='little')
        
        raw_angle = (start_angle + step * i) % 360
        
        final_angle = raw_angle 
        
        if distance > 0 and distance < MAX_DISTANCE:
            points.append((np.radians(final_angle), distance / GRID_TO_MM))
            
    return points

class CoCube:
    def __init__(self, robotID, gateway='192.168.3.1', local_ip='192.168.3.99', ip_prefix=100, udp_port=5000):
        self.robotID = robotID
        self.robot_ip = '.'.join(gateway.split('.')[:-1]) + f".{ip_prefix + robotID}"
        self.robot_port = udp_port + robotID
        self.localIP = local_ip
        self.local_port = udp_port
        try:
            self.sock_listen = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock_listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock_listen.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)  # Send buffer
            self.sock_listen.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)  # Receive buffer
            self.sock_listen.bind((self.localIP, self.robot_port))
            self.sock_listen.settimeout(3)

            self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)  # Send buffer
            self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # Receive buffer
            self.sock_send.settimeout(3)
        except Exception as e:
            print(f"Error: {e}")
            print("Fail to bind IP address，please check whether the IP address is correct！")
        self.pos_p = [0, 0]
        self.pos_m = [0, 0]
        self.yaw = 0
        self.angle = 0
        self.points = deque(maxlen=1000)
        self.uuid_results = {}
        self.running = True
        self.data_process_thread = Thread(target=self.receive_data_thread)
        self.data_process_thread.start()
        print(f"agent_{robotID} initialize successfully!")

    def stop(self):
        self.running = False
        self.data_process_thread.join()
        self.sock_listen.close()
        self.sock_send.close()

    def __del__(self):
        self.stop()

    def receive_data_thread(self):
        while self.running:
            try:
                data, addr = self.sock_listen.recvfrom(1024)
                if not data:
                    continue
                points = parse_ld06(data)
                if points is not None:
                    self.points.extend(points)
                    state = "".join([chr(c) for c in data[PACKET_SIZE:]])
                    x, y, angle = state.split(",")
                    if float(x) > OUTLIER_THRESHOLD or float(y) > OUTLIER_THRESHOLD or float(angle) > 360:
                        print(f"Warning: Outlier detected in position data: x={x}, y={y}, angle={angle}")
                        continue
                    self.pos_p = [float(x), float(y)]
                    self.angle = float(angle)
                    self.yaw = math.radians(float(angle))

            except socket.timeout:
                print(f"No.{self.robotID}: Timeout occurred: No response in receive_positions.")
            except Exception as e:
                print(f"Error: {e}")
                break
        print(f"agent_{self.robotID} thread exit successfully!")

    def get_lidar_points(self):
        points = list(self.points)
        self.points.clear()
        return points

    def send_data(self, str_msg):
        message_bytes = str_msg.encode()
        self.sock_send.sendto(message_bytes, (self.robot_ip, self.local_port))

    def judge_whether_finished(self, uuid):
        if uuid in self.uuid_results.keys():
            result = self.uuid_results[uuid]
            del self.uuid_results[uuid]          # Remove uuids that have already been processed to avoid long dictionary lengths
            return True, result
        else:
            return False, None

    def process_data(self, func, params, block=False, timeout=3):
        '''
        Send the specified functions and parameters to CoCube and return uuid
        Params：
            func: function name
            params: parameters
            block: whether to block
            timeout: timeout
        Returns：
            uuid: unique identifier
        '''
        random_uuid = uuid.uuid4().hex[:6]

        block = str(int(block))
        if params:
            formatted_params = []
            for param in params:
                if isinstance(param, str):
                    formatted_params.append(f'"{param}"')
                else:
                    formatted_params.append(str(param))
            # Send the data to CoCube
            data_to_send = f"{block},{random_uuid},{func}," + ",".join(formatted_params)
        else:
            data_to_send = f"{block},{random_uuid},{func}"
        try:
            self.send_data(data_to_send)
        except Exception as e:
            print(f"Error: {e}")
            print("Fail to send data, please check whether the IP address is correct！")
        
        return random_uuid

     ###### CoCube 感知接口 ########
    
    def get_pos(self):
        return self.pos_p

    def get_pos_m(self):
        return self.pos_m

    def get_angle(self):
        return self.angle

    def get_yaw(self):
        return self.yaw    

    ###### CoCube Control Interface ########
    def move_millisecs(self, direction="forward", speed=40, duration=1000):
        if direction not in ["forward", "backward"]:
            raise ValueError("Direction must be forward or backward")
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        direction = "cocube;" + direction
        return self.process_data(block=True, func="CoCube move for msecs", params=[direction, speed, duration],
                          timeout=3+duration/1000)

    def rotate_millisecs(self, direction="left", speed=40, duration=1000):
        if direction not in ["left", "right"]:
            raise ValueError("Direction must be left or right")
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        direction = "cocube;" + direction
        return self.process_data(block=True, func="CoCube rotate for msecs", params=[direction, speed, duration],
                          timeout=3+duration/1000)

    def move(self, direction="forward", speed=40):
        if direction not in ["forward", "backward"]:
            raise ValueError("Direction must be forward or backward")
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        direction = "cocube;" + direction
        return self.process_data(block=False, func="CoCube move", params=[direction, speed])

    def rotate(self, direction="left", speed=40):
        if direction not in ["left", "right"]:
            raise ValueError("Direction must be left or right")
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        direction = "cocube;" + direction
        return self.process_data(block=False, func="CoCube rotate", params=[direction, speed])
    
    def set_wheel_speed(self, left_speed=40, right_speed=40):
        if not -50 <= left_speed <= 50 or not -50 <= right_speed <= 50:
            raise ValueError("Speed must be in -50 - 50")
        return self.process_data(block=False, func="CoCube set wheel", params=[int(left_speed), int(right_speed)])
    
    def wheels_stop(self):
        return self.process_data(block=False, func="CoCube wheels stop", params=[])

    def wheels_break(self):
        return self.process_data(block=False, func="CoCube wheels break", params=[])
    
    def move_by_steps(self, direction="forward", speed=40, step=50, block=True):
        if direction not in ["forward", "backward"]:
            raise ValueError("Direction must be forward or backward")
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        direction = "cocube;" + direction
        return self.process_data(block=block, func="CoCube move by step", params=[direction, speed, step], timeout=60)
    
    def rotate_by_degree(self, direction="left", speed=40, degree=90, block=True):
        if direction not in ["left", "right"]:
            raise ValueError("Direction must be left or right")
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        direction = "cocube;" + direction
        return self.process_data(block=block, func="CoCube rotate by degree", params=[direction, speed, degree], timeout=100)

    def rotate_to_angle(self, angle=0, speed=40, block=True):
        angle = angle % 360
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        return self.process_data(block=block, func="CoCube rotate to angle", params=[angle, speed])
    
    def point_towards(self, target_x=0, target_y=0, speed=30, block=True):
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        return self.process_data(block=block, func="CoCube point towards", params=[target_x, target_y, speed])

    def move_to_target(self, target_x=0, target_y=0, speed=40, block=True):
        if not 0 <= speed <= 50:
            raise ValueError("Speed must be in 0 - 50")
        return self.process_data(block=block, func="CoCube move to", params=[target_x, target_y, speed])

    ###### CoCube Display Interface ########
    def clear_display(self):
        return self.process_data(block=False, func="[tft:clear]", params=[])

    def set_pixel(self, x=0, y=0, color=(255, 255, 255)):
        if not all(0 <= x <= 240 for x in (x, y)):
            raise ValueError("X and Y must be in 0 - 240")
        if not all(0 <= x <= 255 for x in color):
            raise ValueError("RGB Value must be in 0 - 255")
        color = (color[0] << 16) | (color[1] << 8) | color[2]
        return self.process_data(block=False, func="[tft:setPixel]", params=[x, y, color])

    def fill_rect(self, x=0, y=0, width=10, height=10, color=(255, 255, 255)):
        if not all(0 <= x <= 240 for x in (x, y)):
            raise ValueError("X and Y must be in 0 - 240")
        if not all(0 <= x <= 255 for x in color):
            raise ValueError("RGB Value must be in 0 - 255")
        if not all(0 <= x <= 240 for x in (width, height)):
            raise ValueError("Width and Height must be in 0 - 240")
        color = (color[0] << 16) | (color[1] << 8) | color[2]
        return self.process_data(block=False, func="[tft:rect]", params=[x, y, width, height, color])
    
    def draw_text(self, text="Hello, CoCube!", x=10, y=10, color=(255, 255, 255)):
        if not all(0 <= x <= 240 for x in (x, y)):
            raise ValueError("X and Y must be in 0 - 240")
        if not all(0 <= x <= 255 for x in color):
            raise ValueError("RGB Value must be in 0 - 255")
        color = (color[0] << 16) | (color[1] << 8) | color[2]
        return self.process_data(block=False, func="tft_drawText", params=[text, x, y, color])
    
    def mb_display(self, matrix = np.ones((5, 5)), dtype=int):
        if matrix.shape != (5, 5):
            raise ValueError("Matrix must be 5x5")
        if not np.all((matrix == 0) | (matrix == 1)):
            raise ValueError("Matrix elements must be 0 or 1")
        weights = np.array([2**i for i in range(25)])
        code = matrix.flatten().dot(weights)
        return self.process_data(func="[display:mbDisplay]", params=[code], block="1")

    def set_display_color(self, color=(255, 255, 255)):
        if not all(0 <= x <= 255 for x in color):
            raise ValueError("RGB Value must be in 0 - 255")
        color = (color[0] << 16) | (color[1] << 8) | color[2]
        self.process_data(func="set display color", params=[color], block="0")

    def set_tft_backlight(self, brightness=5):
        if brightness not in range(0, 11):
            raise ValueError("brightness must be 0-10")
        return self.process_data(block=False, func="[tft:setBacklight]", params=[brightness])
        
    def draw_aruco_marker_on_tft(self, id=0):
        if id not in range(0, 100):
            raise ValueError("Aruco Marker ID must be in 0 - 99")
        return self.process_data(block=False, func="CoCube draw Aruco Marker on TFT", params=[id])

    def draw_apriltag_on_tft(self, id=0):
        if id not in range(0, 100):
            raise ValueError("AprilTag ID must be in 0 - 99")
        return self.process_data(block=False, func="CoCube draw AprilTag on TFT", params=[id])

    def call_bmp(self, bmp_name, x=0, y=0):
        return self.process_data(block=False, func="drawBMPfile", params=[bmp_name, x, y])

    def custom_func(self, func="", params=[], block=False):
        return self.process_data(block=block, func=func, params=params)
    
    ###### CoCube External Module Interface ########
    def power_on_module(self):
        return self.process_data(block=False, func="ccmodule_power on module", params=[])

    def gripper_open(self, block=True):
        return self.process_data(block=block, func="ccmodule_gripper open", params=[])

    def gripper_close(self, block=True):
        return self.process_data(block=block, func="ccmodule_gripper close", params=[])

    def gripper_degree(self, degree, block=True):
        if not 0 <= degree <= 70:
            raise ValueError("Degree must be in 0 - 70")
        return self.process_data(block=block, func="ccmodule_gripper degree", params=[degree])

    def set_NeoPixel_color(self, id, color):
        if id not in range(0, 49):
            raise ValueError("NeoPixel ID must be in 1 - 48")
        if not all(0 <= x <= 255 for x in color):
            raise ValueError("RGB Value must be in 0 - 255")
        color = (color[0] << 16) | (color[1] << 8) | color[2]
        return self.process_data(block=False, func="setNeoPixelColor", params=[id, color])

    def set_all_NeoPixels_color(self, color):
        if not all(0 <= x <= 255 for x in color):
            raise ValueError("RGB Value must be in 0 - 255")
        color = (color[0] << 16) | (color[1] << 8) | color[2]
        self.process_data(block=False, func="ccmodule_attach NeoPixels", params=[])
        time.sleep(0.1)
        return self.process_data(block=False, func="ccmodule_set all NeoPixels color", params=[color])

    def clear_NeoPixels(self):
        return self.process_data(block=False, func="clearNeoPixels", params=[])

    def ToF_distance(self):
        self.process_data(block=False, func="ccmodule_ToF connected", params=[])
        time.sleep(0.1)
        return self.process_data(block=True, func="ccmodule_ToF distance", params=[])
