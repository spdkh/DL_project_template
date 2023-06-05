# Copyright 2023 The Improved caGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from data.data import Data
from data.fixed_cell import FixedCell
from utils import fcns


class FairSIM(FixedCell):
    def __init__(self, args):
        Data.__init__(self, args)
        self.data_groups = {'train': 'training',
                            'test': 'testing',
                            'val': 'validation'}

        self.data_types = {'x': 'raw_data', 'y': 'gt'}
        self.args.data_dir = fcns.fix_path(self.args.data_dir)
        input_dir = os.path.join(self.args.data_dir,
                                 self.data_groups['train'],
                                 self.data_types['y'])
        in_sample_dir = os.path.join(input_dir,
                                     os.listdir(input_dir)[0])
        self.input_dim = self.load_sample(in_sample_dir)
        print('input', self.input_dim)
        self.output_dim = [self.input_dim[0] * self.args.scale_factor,
                           self.input_dim[1] * self.args.scale_factor,
                           self.input_dim[2], 1]
        print('output', self.output_dim)

        save_weights_name = 'SIM_Fair'

        self.save_weights_path = os.path.join(self.args.checkpoint_dir,
                                              save_weights_name)

        if not os.path.exists(self.save_weights_path):
            os.mkdir(self.save_weights_path)

        self.sample_path = os.path.join(self.save_weights_path, 'sampled_img')

        self.log_path = os.path.join(self.save_weights_path, 'graph')
        self.data_dirs = dict()

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        for data_group in self.data_groups.keys():
            self.data_dirs[data_group] = os.path.join(self.args.data_dir,
                                                      self.data_groups[data_group])
            for data_type in self.data_types.keys():
                self.data_dirs[data_type + data_group] = \
                    os.path.join(self.data_dirs[data_group],
                                 self.data_types[data_type])
        print(self.data_dirs)
