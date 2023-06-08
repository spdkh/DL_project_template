"""
    author: SPDKH
"""

from src.data.data import Data


class FixedCell(Data):
    """
        FixedCell 3D_SIM Data from Opstad
    """
    def __init__(self, args):
        Data.__init__(self, args)

        self.config()

        self.otf_path = './OTF/fixedcell_otf.tif'
        self.psf = self.init_psf()
