import torch
import torch.nn as nn
import numpy as np


class Propagate(nn.Module):
    def __init__(self,
                 pixelsize,
                 refidx,
                 lambda_,
                 z,
                 NFv, NFh,
                 device,
                 NA = 1,
                 convunit = True,
                 freqmask_fl = True, 
                 FFLimit = True
                 ):
        super(Propagate, self).__init__()
        self.refidx = refidx
        self.freqmask_fl = freqmask_fl
        self.lambda_ = lambda_ * 1e-9 if convunit else lambda_
        self.pixelsize = pixelsize * 1e-6 if convunit else pixelsize
        self.z = z * 1e-6 if convunit else z
        self.NFv, self.NFh = NFv, NFh
        self.NA = NA
        self.device = torch.device(device)
        # modified by zhoutk
        self.FFLimit = FFLimit # limit the angular component for the propagation output, by zhoutk
        self.RFh = self.pixelsize * self.NFh # real size in horizontal direction, by zhoutk
        self.RFv = self.pixelsize * self.NFh
        self.sinMAFh = self.RFh / 2 / np.sqrt(self.RFh ** 2 + self.z ** 2) # limit the diffraction angle to half of the image size(assuming padding=image size)
        self.sinMAFv= self.RFv / 2 / np.sqrt(self.RFv ** 2 + self.z ** 2)
        self.zeroboundarymask = self.zerobound()
        
        ## end modification
        self.init_Matrix()

    def forward(self, img):
        Nv, Nh = img.shape[1:3]
        spectrum = torch.fft.fft2(img, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        spectrum_z = spectrum * self.H.unsqueeze(0)
        if self.freqmask_fl:
            spectrum_z = spectrum_z * self.freqmask
        if self.FFLimit: # limit angular component for the propagation output, by zhoutk
            spectrum_z = spectrum_z * self.lamask
        spectrum_z = torch.fft.ifftshift(spectrum_z, dim=(-2, -1))
        img_z = torch.fft.ifft2(spectrum_z, dim=(-2, -1))
        img_z = img_z[:, :Nv, :Nh]
        img_z = img_z * torch.tensor(self.zeroboundarymask, dtype=torch.complex64, device=self.device)
        return img_z
    
    
    # def AngLimit(self, spectrum_z): # remove large and small angle diffraction component
    #     #spectrum_z = spectrum_z * self.lamask # large angle
    #     PMNh = int(self.sinMAFh/(1/self.NFh)) #Maximal frequency pixels permitted by propagation distances
    #     PMNv = int(self.sinMAFv/(1/self.NFv)) 
    #     img_z = torch.fft.ifft2(spectrum_z, dim=(-2, -1))
    #     return img_z

    def init_Matrix(self):
        Fs = 1 / self.pixelsize
        Fh = Fs / self.NFh * np.arange((-np.ceil((self.NFh - 1) / 2)), np.floor((self.NFh - 1) / 2) + 0.5)
        # Fv = Fs / self.NFv * np.arange((-np.ceil((self.NFv - 1) / 2)), np.floor((self.NFv - 1) / 2) + 0.5)
        Fv = 0
        [Fhh, Fvv] = np.meshgrid(Fh, Fv)
        np_H = self.PropGeneral(Fhh, Fvv)
        np_freqmask = self.BandLimitTransferFunction(Fvv, Fhh)
        # na
        na_freqmask = self.NALimit(Fvv, Fhh)
        np_freqmask = np_freqmask & na_freqmask
        self.H = torch.tensor(np_H, dtype=torch.complex64, device=self.device)
        self.freqmask = torch.tensor(np_freqmask, dtype=torch.complex64, device=self.device)
        self.lamask = torch.tensor(self.LargeAngleLimit(Fvv, Fhh), device=self.device)

    def PropGeneral(self, Fhh, Fvv):
        DiffLimMat = np.ones(Fhh.shape)
        lamdaeff = self.lambda_ / self.refidx
        DiffLimMat[(Fhh ** 2.0 + Fvv ** 2.0) >= (1.0 / lamdaeff ** 2.0)] = 0.0

        temp1 = 2.0 * np.pi * self.z / lamdaeff
        temp3 = (lamdaeff * Fvv) ** 2.0
        temp4 = (lamdaeff * Fhh) ** 2.0
        temp2 = np.complex128(1.0 - temp3 - temp4) ** 0.5
        H = np.exp(1j * temp1 * temp2)
        H[np.logical_not(DiffLimMat)] = 0
        return H

    def BandLimitTransferFunction(self, Fvv, Fhh):
        hSize, vSize = Fvv.shape
        dU = (hSize * self.pixelsize) ** -1.0
        dV = (vSize * self.pixelsize) ** -1.0
        Ulimit = ((2.0 * dU * self.z) ** 2.0 + 1.0) ** -0.5 / self.lambda_
        Vlimit = ((2.0 * dV * self.z) ** 2.0 + 1.0) ** -0.5 / self.lambda_
        freqmask = ((Fvv ** 2.0 / (Ulimit ** 2.0) + Fhh ** 2.0 * (self.lambda_ ** 2.0)) <= 1.0) & \
                   ((Fvv ** 2.0 * (self.lambda_ ** 2.0) + Fhh ** 2.0 / (Vlimit ** 2.0)) <= 1.0)
        return freqmask

    def NALimit(self, Fvv, Fhh):
        freq_limit = 1 / self.lambda_ * self.NA
        namask = (Fvv ** 2.0 + Fhh ** 2.0) < freq_limit ** 2
        return namask
    
    def LargeAngleLimit(self, Fvv, Fhh): # added by tk remove the large angle diffraction 
        
        freq_limit_h = 1 / self.lambda_ * self.sinMAFh
        freq_limit_v = 1 / self.lambda_ * self.sinMAFv
        lamask = np.logical_and(np.abs(Fvv) < freq_limit_v  , np.abs(Fhh) < freq_limit_h )
        return lamask

    def zerobound(self):
        zeroboundarymask = np.zeros((1, self.NFv, self.NFh))
        zeroboundarymask[0, :, self.NFh//2-self.NFh//4:self.NFh//2+self.NFh//4] = 1
        return zeroboundarymask

class PropagateList(nn.Module):
    def __init__(self,
                 pixelsize,
                 refidx,
                 lambda_,
                 z_list,
                 NFv, NFh,
                 device,
                 NA=1,
                 convunit=True,
                 freqmask_fl=True,
                 FFLimit=False,
                 ):
        super(PropagateList, self).__init__()
        self.model = nn.ModuleList([])
        for z in z_list:
            self.model.append(Propagate(pixelsize, refidx, lambda_, z, NFv, NFh, device, NA, convunit, freqmask_fl, FFLimit))

    def forward(self, x):
        x_log = []
        for layer in self.model:
            x = layer(x)
            x_log.append(x.clone())
        return x, x_log
