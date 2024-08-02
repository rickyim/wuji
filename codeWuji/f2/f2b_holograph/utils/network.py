import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.propagation import Propagate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

class Scan_ONN(nn.Module):
    def __init__(self,
                 mask_rc,
                 tile_rc,
                 cell_rc,
                 device,
                 encode_matrix,
                 decode_matrix,
                 Prokw_Multi,
                 Prokw,
                 propad,
                 EncodeFocus,
                 DecodeFocus):
        super(Scan_ONN, self).__init__()
        self.mask_row, self.mask_col = mask_rc
        self.tile_row, self.tile_col = tile_rc
        self.cell_row, self.cell_col = cell_rc
        self.mask_pixel_num = self.mask_row * self.mask_col
        self.Propagate = Propagate(**Prokw)
        self.propad = propad
        self.ProPad = nn.ZeroPad2d(propad)
        self.device = torch.device(device)
        self.encode_matrix = torch.ones((1, self.tile_row, self.tile_col, 1), device=self.device)
        self.encode_co = torch.ones((1, self.tile_row, self.tile_col, 1), device=self.device)
        
        self.mask_voltage = nn.Parameter(torch.rand(1, self.mask_pixel_num, device=self.device))
        self.Propagate_multi = []
        self.voltage_to_phase = []
        self.phase_conpensate = []
        self.lambda_ = [1530, 1540, 1550, 1560]
        for lamda in self.lambda_:
            Prokw['lambda_'] = lamda
            self.Propagate_multi.append(Propagate(**Prokw))
        
    def forward(self, img, iteration):
        
        measurement = []
        mask_voltage = self.mask_voltage % 2.5 + 0.5

        for idx, lamda in enumerate(self.lambda_):
            phase = self.voltage_to_phase[idx][:, 1] * mask_voltage * mask_voltage + self.voltage_to_phase[idx][:, 2] - self.phase_conpensate[idx]
            mask = torch.complex(torch.cos(phase), torch.sin(phase))
            hidden = img * mask
            hidden = self.input_encoding(hidden)
            hidden = hidden.reshape((-1, self.mask_row * self.cell_row, self.mask_col * self.cell_col))
            hidden = F.pad(hidden, (self.propad, self.propad), 'constant', 0)
            hidden = self.Propagate_multi[idx](hidden)
            hidden = hidden[:, :, self.propad:-self.propad]
            measurement.append(torch.square(torch.abs(hidden)))
        measurement = torch.cat(measurement, dim=1)
                
        return measurement


    def input_encoding(self, input):
        input = torch.reshape(input, (-1, 1, 1, self.mask_row * self.mask_col))
        input_tile = input * self.encode_matrix
        input_tile = input_tile * self.encode_co

        input_paddings = F.pad(input_tile,
                               (0, 0,
                                int((self.cell_col - self.tile_col) / 2), int((self.cell_col - self.tile_col) / 2),
                                int((self.cell_row - self.tile_row) / 2), int((self.cell_row - self.tile_row) / 2),
                                0, 0),
                               mode='constant',
                               value=0.0)
        input = input_paddings.reshape(-1, self.cell_row, self.cell_col, self.mask_row, self.mask_col)
        input = torch.permute(input, (0, 3, 1, 4, 2))
        input = input.reshape((-1, self.mask_row * self.cell_row, self.mask_col * self.cell_col))
        return input

    def output_decoding(self, output):
        output = output.reshape((-1, self.mask_row, self.cell_row, self.mask_col, self.cell_col))
        output = torch.permute(output, (0, 2, 4, 1, 3))
        output = output[:,
                 int((self.cell_row - self.tile_row) / 2):self.tile_row + int((self.cell_row - self.tile_row) / 2),
                 int((self.cell_col - self.tile_col) / 2):self.tile_col + int((self.cell_col - self.tile_col) / 2),
                 :, :]

        # mean
        output = torch.square(torch.abs(output))
        output = output.sum(dim=(1, 2))
        output = output.reshape((-1, self.mask_row * self.mask_col))
        return output
