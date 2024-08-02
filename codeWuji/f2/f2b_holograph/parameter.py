import os
import torch
import random


class Params(object):
    def __init__(self):
        self.exp_id = 'Test'
        self.manual_seed = random.randint(1, 10000)
        self.use_cuda = True
        self.device = 'cuda'
        torch.cuda.manual_seed_all(self.manual_seed)
        self.core = {
            'train': {
                'lr': 0.1, 'batch_size': 32, 'max_epoch': 2000,
                'validate_iter': 1, 'save_checkpoint_iter': 1,
                'BaseFile': None,
                'CheckPointFile': None
            },
            'network': {
                'mask_rc': [1, 128], 'tile_rc': [1, 4], 'cell_rc': [1, 10],
                'device': self.device,
                'encode_matrix': './utils/encode0811_1.mat',
                'decode_matrix': './utils/encode0811_1.mat',
                'Prokw_Multi': {
                    'pixelsize': 4.0, 'refidx': 1.0, 'lambda_': 1550,
                    'z_list': [2e5] * 5, 'device': self.device, 'convunit': True,
                    'freqmask_fl': True, 'FFLimit': True,
                },
                'Prokw': {
                    'pixelsize': 4.0, 'refidx': 1.0, 'lambda_': 1550,
                    'z': 0.5e6, 'device': self.device, 'convunit': True,
                    'freqmask_fl': True, 'FFLimit': True,
                },
                'propad': 25600,
                'EncodeFocus': {
                    'f_type': 0, 'f_mode': 1, 'focal_len': 640,
                    'Prokw':{
                        'pixelsize': 4.0, 'refidx': 1.0, 'lambda_': 1550,
                        'z': 640, 'device': self.device, 'convunit': True,
                        'freqmask_fl': True}
                },
                'DecodeFocus': {
                    'f_type': 1, 'f_mode': 0, 'focal_len': None,
                    'Prokw':{
                        'pixelsize': 4.0, 'refidx': 1.0, 'lambda_': 1550,
                        'z': 80, 'device': self.device, 'convunit': True,
                        'freqmask_fl': True},
                },
                
            }
        }
        self.init_sync()
        self.root_file = './record'
        self.summary_dir = os.path.join(self.root_file, 'log', self.exp_id)
        self.checkpoint_dir = os.path.join(self.root_file, 'state', self.exp_id)
        print('exp_id: ', self.exp_id)
        
    def init_sync(self):
        mr, mc = self.core['network']['mask_rc'][0], self.core['network']['mask_rc'][1]
        cr, cc = self.core['network']['cell_rc'][0], self.core['network']['cell_rc'][1]
        pp = self.core['network']['propad']
        self.core['network']['Prokw']['NFv'] = mr * cr #+ ppr * 2
        self.core['network']['Prokw']['NFh'] = mc * cc + pp * 2
        self.core['network']['Prokw_Multi']['NFv'] = mr * cr #+ ppr * 2
        self.core['network']['Prokw_Multi']['NFh'] = mc * cc + pp * 2
        self.core['network']['EncodeFocus']['Prokw']['NFv'] = mr * cr #+ ppr * 2
        self.core['network']['EncodeFocus']['Prokw']['NFh'] = mc * cc + pp * 2
        self.core['network']['DecodeFocus']['Prokw']['NFv'] = mr * cr #+ ppr * 2
        self.core['network']['DecodeFocus']['Prokw']['NFh'] = mc * cc + pp * 2
        self.core['network']['EncodeFocus']['propad'] = pp
        self.core['network']['EncodeFocus']['mask_row'] = mr
        self.core['network']['EncodeFocus']['mask_col'] = mc
        self.core['network']['EncodeFocus']['cell_row'] = cr
        self.core['network']['EncodeFocus']['cell_col'] = cc
        self.core['network']['DecodeFocus']['propad'] = pp
        self.core['network']['DecodeFocus']['mask_row'] = mr
        self.core['network']['DecodeFocus']['mask_col'] = mc
        self.core['network']['DecodeFocus']['cell_row'] = cr
        self.core['network']['DecodeFocus']['cell_col'] = cc


if __name__ == '__main__':
    p = Params()
    pass
