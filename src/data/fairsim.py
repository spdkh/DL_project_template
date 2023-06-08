"""
Copyright 2023 The Improved caGAN Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from scipy.io import loadmat

from src.data.data import Data
from src.utils.physics_informed.psf_generator import Parameters3D, cal_psf_3d, psf_estimator_3d


class FairSIM(Data):
    """
        Manage FairSIM data Class
    """
    def __init__(self, args):
        Data.__init__(self, args)

        self.data_types = {'x': 'raw_data', 'y': 'gt'}

        self.config()
        self.otf_path = './OTF/splinePSF_128_128_11.mat'
        # self.psf = self.init_psf()

    def load_psf(self):
        raw_psf = loadmat(self.otf_path)
        raw_psf = raw_psf['h']
        print(np.shape(raw_psf))
        return raw_psf
