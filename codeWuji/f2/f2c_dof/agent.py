import cv2
import json
import pickle
from parameter import Params
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# from utils.network import ONN
from utils.network import ONN
from utils.utils import *
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class BaseAgent:
    def __init__(self, params):
        self.criterion = nn.MSELoss(reduction='sum')
        self.params = params
        self.batch_size = self.params.core['train']['batch_size']
        self.manual_seed = self.params.manual_seed
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.Dataset = Dataset(params.dataset_name, split='train')
        params.core['network']['num_wavelength'] = 16
        self.network = ONN(**params.core['network'])
        self.update_settings()

    def update_settings(self):
        self.current_epoch = 0
        self.current_iteration = 0
        if params.core['train']['BaseFile'] is not None:
            filename = params.core['train']['BaseFile']
            checkpoint = torch.load(filename)['network']
            self.network.load_state_dict(checkpoint, strict=False)
        self.device = torch.device(params.device)
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.network.to(self.device)
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir, comment='biscale')
        self.lr = self.params.core['train']['lr']
        self.optimizer = torch.optim.Adam([{
            "params": filter(lambda p: p.requires_grad, self.network.parameters()),
            "lr": self.params.core['train']['lr'],
            "weight_decay": 0
        }])
        self.load_checkpoint(filename=self.params.core['train']['CheckPointFile'])

    def train(self):
        os.makedirs(os.path.join(self.params.checkpoint_dir, ''), exist_ok=True)
        with open(os.path.join(self.params.checkpoint_dir, 'params.pth'), 'wb') as fp:
            pickle.dump(self.params, fp)
        with open(os.path.join(self.params.checkpoint_dir, 'core.json'), 'w') as fp:
            json.dump(self.params.core, fp)
        print('start training')
        Dataset = DataLoader(self.Dataset, batch_size=self.params.core['train']['batch_size'], shuffle=True,
                             num_workers=2)
        for epoch in range(self.current_epoch, self.params.core['train']['max_epoch']):
            self.current_epoch = epoch
            self.train_one_epoch(Dataset)

    def train_one_epoch(self, Dataset):
        torch.manual_seed(72)
        self.grating_num = 128
        self.train_epochs = 2000
        self.random_batch = 100
        self.num_wavelength = params.core['network']['num_wavelength']
        result_loss = np.zeros((self.grating_num, self.random_batch, self.num_wavelength))
        for idx in range(self.grating_num):
            tqdm.write('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            n = idx + 1
            average_loss = torch.zeros(self.random_batch, self.num_wavelength).to(self.device)

            for rand_num in range(self.random_batch):
                self.ground_truth = torch.rand((self.num_wavelength, 1, n))
                self.gt_norm = torch.tensor(np.linalg.norm(self.ground_truth, ord=2, axis=2, keepdims=True))
                self.ground_truth = self.ground_truth / self.gt_norm # value to one
                log = AverageMeterList(2)
                iteration = tqdm(Dataset, leave=False, desc='Train', ncols=120)
                loss_min = 10000
                average_loss_min = torch.ones(self.num_wavelength).to(self.device)

                for epoch, samples in enumerate(iteration):
                    torch.cuda.empty_cache()
                    sample = torch.ones((1, self.grating_num))
                    sample = sample.to(self.device) 
                    label = self.ground_truth.to(self.device)
                    self.optimizer.zero_grad()
                    self.network.zero_grad()
                    output, couple = self.network(sample, label)
                    output = F.interpolate(output, size=n, mode='nearest')
                    output = output / torch.norm(output, p=2, dim=2, keepdim=True)
                    loss = self.criterion(output, label)
                    loss.backward()
                    self.optimizer.step()
                    if loss.cpu().detach() < loss_min:
                        loss_min = loss.cpu().detach()
                        for wave in range(self.num_wavelength):           
                            average_loss_min[wave] = self.criterion(output[wave], label[wave]).sqrt()
                    log.update([loss.cpu().detach().numpy(), couple.squeeze().cpu().detach().numpy()])
                    self.current_iteration += 1
                    iteration.set_postfix_str(f'{log}')
                    if epoch == self.train_epochs:
                        average_loss[rand_num] = average_loss_min.detach()
                        break
                     
                loss_avg, couple_avg = log.avg()
                self.summary_writer.add_scalar("epoch/loss", loss_avg, self.current_epoch)
                self.summary_writer.add_scalar("epoch/energy", couple_avg, self.current_epoch)
            tqdm.write(f'Train grating num {n} epoch: {log}')
            print(average_loss.cpu().detach().mean())
            result_loss[idx] = average_loss.cpu().detach()


if __name__ == '__main__':
    
    params = Params()
    agent = BaseAgent(params)
    agent.train()
