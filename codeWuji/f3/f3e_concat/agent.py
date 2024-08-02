import cv2
from PIL import Image
from utils.utils import *
from utils.network import Scan_ONN
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from parameter import Params
import pickle
import json
import os
matplotlib.use('Agg')


class BaseAgent:
    def __init__(self, params):

        self.distance = 1000
        self.concate = 16
        self.N = 1
        self.chip_concate_num = self.N * (self.concate - 1) + self.concate
        params.core['network']['mask_rc'] = [1, 128 * self.chip_concate_num]
        params.core['network']['Prokw']['z'] = self.distance * 1e6
        params.init_sync()

        self.criterion = MyLoss()
        self.params = params
        self.manual_seed = self.params.manual_seed
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.Dataset_T = Dataset(params.dataset_name, split='train')
        self.network = Scan_ONN(**params.core['network'])
        self.update_settings()
        self.mask_pixel_num = params.core['network']['mask_rc'][1]

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
        self.summary_writer = SummaryWriter(
            log_dir=self.params.summary_dir, comment='biscale')
        self.lr = self.params.core['train']['lr']
        self.optimizer = torch.optim.Adam([{
            "params": filter(lambda p: p.requires_grad, self.network.parameters()),
            "lr": self.params.core['train']['lr'],
            "weight_decay": 0
        }])
        self.load_checkpoint(
            filename=self.params.core['train']['CheckPointFile'])

    def train(self):
        os.makedirs(os.path.join(
            self.params.checkpoint_dir, ''), exist_ok=True)
        with open(os.path.join(self.params.checkpoint_dir, 'params.pth'), 'wb') as fp:
            pickle.dump(self.params, fp)
        with open(os.path.join(self.params.checkpoint_dir, 'core.json'), 'w') as fp:
            json.dump(self.params.core, fp)
        print('start training')
        Dataset = DataLoader(self.Dataset_T, batch_size=self.params.core['train']['batch_size'], shuffle=True,
                             num_workers=2)
        for epoch in range(self.current_epoch, self.params.core['train']['max_epoch']):
            self.current_epoch = epoch
            self.train_scan_one_epoch(Dataset)


    def train_scan_one_epoch(self, Dataset):

        for iter in range(608):
            log = AverageMeterList(4)
            iteration = tqdm(Dataset, leave=False, desc=f'Train {self.current_epoch} epoch', ncols=120)
            matrix = experiment_matrix[iter]
            print('---------- iteration {} ------------'.format(iter))
            for _, samples in enumerate(iteration):
                torch.cuda.empty_cache()
                sample, label = samples
                self.batch_size = sample.shape[0]
                sample = torch.ones((self.batch_size, 1, self.mask_pixel_num)).to(self.device)
                
                self.optimizer.zero_grad()
                self.network.zero_grad()
                output = self.network(sample, self.N, self.concate)
                ground_truth = torch.zeros(output.shape, device=self.device)
                self.lambda_ = [1530, 1540, 1550, 1560]
                exp_holo = matrix
                data_without_0 = []
                for lambda_idx in range(len(self.lambda_)):
                    for i in range(4):
                        ground_truth[:, lambda_idx, 1400 + 5000 * (i)] = (exp_holo[lambda_idx][i] == 1) * 1 + (exp_holo[lambda_idx][i] == 0) * -1
                    if exp_holo[lambda_idx].sum() != 0:
                        data_without_0.append(lambda_idx)
                        ground_truth[:, lambda_idx] = ground_truth[:, lambda_idx] / torch.sum(ground_truth[0][lambda_idx][torch.where(ground_truth[0][lambda_idx] > 0)]) * 128 * 0.01
                loss = F.mse_loss(output[:, np.array(data_without_0)], ground_truth[:, np.array(data_without_0)])

                loss.backward()
                self.optimizer.step()

                log.update([loss.cpu().detach().numpy(), loss.cpu().detach().numpy(),
                            loss.cpu().detach().numpy(), loss.cpu().detach().numpy()])
                self.current_iteration += 1
                iteration.set_postfix_str(f'{log}')


if __name__ == '__main__':
    params = Params()
    agent = BaseAgent(params)
    agent.train()
