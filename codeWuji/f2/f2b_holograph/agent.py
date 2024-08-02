from utils.utils import *
from utils.network import Scan_ONN
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from parameter import Params
import pickle
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
matplotlib.use('Agg')


class BaseAgent:
    def __init__(self, params):
        self.criterion = MyLoss()
        self.params = params
        self.manual_seed = self.params.manual_seed
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.Dataset_T = Dataset(params.dataset_name, split='train')
        self.Dataset_V = Dataset(params.dataset_name, split='val')
        self.network = Scan_ONN(**params.core['network'])
        self.update_settings()
        self.mask_pixel_num = 128

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
        Dataset_test = DataLoader(self.Dataset_V, batch_size=self.params.core['train']['batch_size'], shuffle=True,
                                  num_workers=2)
        for epoch in range(self.current_epoch, self.params.core['train']['max_epoch']):
            self.current_epoch = epoch
            self.train_scan_one_epoch(Dataset)
            if epoch % self.params.core['train']['save_checkpoint_iter'] == 0:
                self.save_checkpoint()
            if epoch % self.params.core['train']['validate_iter'] == 0:
                torch.cuda.empty_cache()
                self.validate_scan(Dataset_test)

    def train_scan_one_epoch(self, Dataset):
        log = AverageMeterList(4)
        iteration = tqdm(Dataset, leave=False, desc=f'Train {self.current_epoch} epoch', ncols=120)
        for _, samples in enumerate(iteration):
            
            if _ > 300: break
            
            torch.cuda.empty_cache()
            sample, label = samples
            self.batch_size = sample.shape[0]
            sample = torch.ones((self.batch_size, 1, self.mask_pixel_num)).to(self.device)
            
            self.optimizer.zero_grad()
            self.network.zero_grad()
            output = self.network(sample, _)
            ground_truth = torch.zeros(output.shape, device=self.device)

            self.lambda_ = [1530, 1540, 1550, 1560]
            exp_holo = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1],])
            exp_holo = (exp_holo > 0) * 1 + (exp_holo == 0) * 0

            for lambda_idx in range(len(self.lambda_)):
                for i in range(4):
                    bias = -200
                    ground_truth[:, lambda_idx, -bias + 120 + 200 * (i)] = exp_holo[lambda_idx][i]
                ground_truth[:, lambda_idx] = ground_truth[:, lambda_idx] / torch.sum(ground_truth[0][lambda_idx][torch.where(ground_truth[0][lambda_idx] > 0)]) * 128 * 0.20

            loss = F.mse_loss(output, ground_truth)
            loss.backward()
            self.optimizer.step()

            log.update([loss.cpu().detach().numpy(), loss.cpu().detach().numpy(),
                        loss.cpu().detach().numpy(), loss.cpu().detach().numpy()])
            self.current_iteration += 1
            iteration.set_postfix_str(f'{log}')
            

    def validate_scan(self, Dataset_test):
        tqdm.write('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        log = AverageMeterList(4)
        iteration = tqdm(Dataset_test, leave=False, desc='Val', ncols=120)
        for _, samples in enumerate(iteration):
            torch.cuda.empty_cache()
            sample, label = samples
            self.batch_size = sample.shape[0]
            sample = torch.ones((self.batch_size, 1, self.mask_pixel_num)).to(self.device)
            
            output = self.network(sample)


    def save_checkpoint(self, is_best=False):
        file_name = 'epoch_%s.pth.tar' % str(self.current_epoch).zfill(5)
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(self.params.checkpoint_dir, file_name))

    def load_checkpoint(self, filename):
        if filename is None:
            print('do not load checkpoint')
        else:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.network.load_state_dict(checkpoint['network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
    def plot_log(self, hidden_log):
        plt.figure(figsize=(12, 10))
        for i, hid in enumerate(hidden_log):
            h = hid[0].abs().cpu().detach().numpy()
            plt.subplot(1, 3, i + 1)
            #plt.imshow(h / np.max(h))
            plt.plot(np.squeeze(h))
            plt.title(f'{i + 1}')
        plt.savefig(os.path.join(self.params.checkpoint_dir,
                    'epoch_%s.png' % str(self.current_epoch).zfill(5)))
        plt.close()


if __name__ == '__main__':
    params = Params()
    agent = BaseAgent(params)
    agent.train()
