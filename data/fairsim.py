"""
    author: SPDKH
"""
import os

import numpy as np
from scipy.io import loadmat

from data.data import Data
from utils import fcns
from utils.psf_generator import Parameters3D, cal_psf_3d, psf_estimator_3d


class FairSIM(Data):
    def __init__(self, args):
        Data.__init__(self, args)
        self.data_groups = {'train': 'training',
                            'test': 'testing',
                            'val': 'validation'}

        self.data_types = {'x': 'raw_data', 'y': 'gt'}
        self.args.data_dir = fcns.fix_path(self.args.data_dir)
        input_dir = os.path.join(self.args.data_dir,
                                 self.data_groups['train'],
                                 self.data_types['x'])
        output_dir = os.path.join(self.args.data_dir,
                                  self.data_groups['train'],
                                  self.data_types['y'])
        in_sample_dir = os.path.join(input_dir,
                                     os.listdir(input_dir)[0])
        self.input_dim = self.load_sample(in_sample_dir)
        print('input', self.input_dim)
        out_sample_dir = os.path.join(output_dir,
                                      os.listdir(output_dir)[0])

        self.output_dim = self.load_sample(out_sample_dir)
        print('output', self.output_dim)

        for data_group in self.data_groups.keys():
            self.data_dirs[data_group] = os.path.join(self.args.data_dir,
                                                      self.data_groups[data_group])
            for data_type in self.data_types.keys():
                self.data_dirs[data_type + data_group] = \
                    os.path.join(self.data_dirs[data_group],
                                 self.data_types[data_type])

        print(self.data_dirs)
        self.otf_path = './OTF/splinePSF_128_128_11.mat'

        self.psf = self.init_psf()

    def load_psf(self):
        raw_psf = loadmat(self.otf_path)
        raw_psf = raw_psf['h']
        print(np.shape(raw_psf))
        return raw_psf
