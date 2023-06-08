"""
    author: SPDKH
"""
import datetime
import os
from src.data.data import Data
from src.utils import const


class FixedCell(Data):
    def __init__(self, args):
        Data.__init__(self, args)

        self.config()

        self.otf_path = './OTF/fixedcell_otf.tif'
        self.psf = self.init_psf()
