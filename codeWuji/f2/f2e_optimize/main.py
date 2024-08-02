import serial
import serial.tools.list_ports
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
from laser import TSL

class VoltageOperator(object):
    def __init__(self):
        # connect serial
        self.ser = serial.Serial(port="COM5",
                            baudrate=115200,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            timeout=0.5)
        if self.ser.isOpen():                 
            print("Success")
            print(self.ser.name)
        else:
            print("Failed")

    def sendVoltageToDev(self, voltages):

        data =['EB', '90', '00', '00']
        voltage_list = np.array(voltages, dtype=np.uint8)
        sum = 0
        for ii in range(128):
            hex_data = hex(voltage_list[ii])
            sum += voltage_list[ii]
            data.append(hex_data[2:].zfill(2))
        sum += 123
        data.append("{:#04X}".format(sum % 256)[2:])
        data.append('AA')

        hex_str = ' '.join(data)
        send_data = bytes.fromhex(hex_str)
        write_len = self.ser.write(send_data)

        return send_data

    def close(self):
        self.ser.close()

def get_intensity(file_name):
    img = cv2.imread(file_name)
    if img is None:
        time.sleep(0.3)
        img = cv2.imread(file_name)
    img = img[:,:,0]
    size_y, size_x = img.shape
    intensity_value = np.zeros(size_x)
    for i in range(0, size_x):
        value = 0
        for j in range(size_y):
            value += img[j][i]
        intensity_value[i] = value
    return intensity_value

def scan_vertical_phase(lambda_, theta, size):

    if theta == 0:
        mask = np.zeros((size))
    else:
        img_size_x, img_size_y = size, 1
        pixelsize = 40 * 1e-6
        lambda_ = lambda_ * 1e-9
        
        interval = pixelsize / lambda_ * theta * np.pi * 2
        mask = np.arange(0, interval * img_size_x, interval)
        mask = mask[0:size] 
    return mask

def scan_parallel_phase(lambda_, target_depth, size):
    c_SIZE = [1, size]
    lmbda = lambda_ * 1e-9
    K = 2 * np.pi / lmbda
    f = target_depth
    pixelsize = 40e-6

    z = np.arange(0,c_SIZE[0]) - c_SIZE[0]//2
    x = np.arange(0,c_SIZE[1]) - c_SIZE[1]//2
    x, z = np.meshgrid(x * pixelsize, z * pixelsize)
    E = - (2 * np.pi / (2 * lmbda * f)) * (x**2 + z**2)
    mask = E
    
    return mask.squeeze()

def count_ones_in_binary(decimal_number):
    binary_str = bin(decimal_number)[2:]
    count_ones = 0
    for bit in binary_str:
        if bit == '1':
            count_ones += 1
    return count_ones

def int_to_bin_array(value):
    bin_str = format(value, '08b')
    bin_array = np.array(list(bin_str), dtype=int)
    return bin_array

if __name__ == '__main__':

    laser = TSL()
    laser.set_state(1)
    laser.set_wav(1550)
    laser.set_power_dBm(13)
    time.sleep(0.3)

    inst = VoltageOperator()

    volts = list()
    for i in range(128):
        volts.append(int(0000))

    # load parameters
    bias = 0
    num = 128
    lambda_ = 1554.13

    # generate phase
    mask_conpensate = scan_parallel_phase(lambda_=lambda_, target_depth=0.372, size=num) - scan_vertical_phase(lambda_, theta=0.00048, size=128)
    mask_p = scan_parallel_phase(lambda_=lambda_, target_depth=0.50, size=num)

    for group in range(256):

        weight = int_to_bin_array(group)
        if count_ones_in_binary(group) == 1: laser.set_power_dBm(10); time.sleep(0.2)
        else: laser.set_power_dBm(13); time.sleep(0.2)

        # each group for one holograph
        min_error_1 = 1
        min_error_0 = 1
        lr = 0.1
        standard = 0.005
        for t in range(10000):
            if 100 < t and t <= 200: standard = 0.005
            elif 200 < t and t <= 300: standard = 0.005
            elif 300 < t and t <= 400: standard = 0.005
            if t > 400: break

            random_phase = np.random.randn(num) * lr
            # send voltage
            for i in range(bias, bias+num):
                phase = - mask_conpensate[i] + mask_phase[i] + random_phase[i]
                # conpensate
                phase = np.mod(phase, np.pi*2)
                phase = np.uint(phase / (2 * np.pi) * 256)
                vol = phase2vol[i][phase]
                volts[i] = int(vol)
            bytes_data = inst.sendVoltageToDev(volts)
            time.sleep(0.20)

            file_names = os.listdir('./picture')
            sorted_file_names = sorted(file_names)
            data = get_intensity(file_name='./picture/%s'%sorted_file_names[-2])

            # deal with peak
            data = data[::-1]
            peak_idx = [53, 130, 208, 282, 365, 440, 517, 594]
            peak = []
            for i in range(8):
                if weight[i] == 1:
                    peak.append(np.max(data[peak_idx[i]-20:peak_idx[i]+20]))
                else:
                    peak.append(np.min(data[peak_idx[i]-10:peak_idx[i]+10]))
            peak = np.array(peak)
            value_1 = peak[np.where(weight == 1)]
            value_0 = peak[np.where(weight == 0)]
            error_1 = (value_1.max() - value_1.min()) / (value_1.max() + value_1.min())
            print("******************************** group:%d lr:%.2f idx:%d"%(group, lr, t))
            print("error 1: %.2f%%"%(100*error_1) + "minimum error 1: %.2f%%"%(100*min_error_1))
            error_0 = (value_0.max()) / (value_1.max() + value_1.min()) * 2
            print("error 0: %.2f%%"%(100*error_0) + "minimum error 0: %.2f%%"%(100*min_error_0))

            if min_error_1 >= 0.04:
                if error_1 < min_error_1:
                    min_error_1 = error_1
                    min_error_0 = error_0
                    mask_phase = mask_phase + random_phase
            else:
                if min_error_0 >= standard:
                    if error_0 < min_error_0 and error_1 < 0.04:
                        min_error_1 = error_1
                        min_error_0 = error_0
                        mask_phase = mask_phase + random_phase
                else:
                    if min_error_1 > 0.035:
                        lr = 0.05
                    else:
                        lr = 0.02
                    if error_1 < min_error_1 and error_0 < standard:
                        min_error_1 = error_1
                        min_error_0 = error_0
                        mask_phase = mask_phase + random_phase

        print("minimum error 1: %.2f%%"%(100*min_error_1))
        print("minimum error 0: %.2f%%"%(100*min_error_0))
    inst.close()