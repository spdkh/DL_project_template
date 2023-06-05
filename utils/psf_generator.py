"""
    caGAN project psf
"""
import math
import numpy as np
import numpy.fft as F
from skimage.metrics import mean_squared_error as compare_mse, \
                            normalized_root_mse as compare_nrmse,\
                            peak_signal_noise_ratio as compare_psnr,\
                            structural_similarity as compare_ssim
from numpy import asarray as ar, exp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from utils.read_mrc import read_mrc
import tifffile as tiff


class Parameters3D:
    """
        3D PSF Parameters
    """
    # set the Parameters3D for 3D SIM
    def __init__(self):
        """
            Initial Parameters
        """
        self.lamda = 525e-3
        self.Nx = 128
        self.Ny = 128
        self.Nz = 11
        self.dx = 61e-3 / 2
        self.dy = 61e-3 / 2
        self.dz = 160e-3
        self.dxy = self.dx
        self.dkx = 1 / (self.Nx * self.dx)
        self.dky = 1 / (self.Ny * self.dy)
        self.dkz = 1 / (self.Nz * self.dz)
        self.dkr = np.min([self.dkx, self.dky])
        self.nphases = 5
        self.ndirs = 3
        self.NA = 1.1
        self.NAEx = 0.5
        self.nimm = 1.406
        self.space = self.lamda / self.NAEx / 2
        self.k0mod = 1 / self.space
        self.norders = int((self.nphases + 1) / 2)


def prctile_norm(x, min_prc=0, max_prc=100):
    """
    :param x:
    :param min_prc:
    :param max_prc:
    :return:
    """
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y


def cal_psf_3d(raw_psf, dim):
    """
    :param otf_path:
    :param Ny:
    :param Nx:
    :param Nz:
    :param dky:
    :param dkx:
    :param dkz:
    :return:
    """
    curOTF = None
    PSF = None
    Ny, Nx, Nz = dim

    dx = 61e-3 / 2
    dy = 61e-3 / 2
    dz = 160e-3
    dkx = 1 / (Nx * dx)
    dky = 1 / (Ny * dy)
    dkz = 1 / (Nz * dz)

    diagdist = math.ceil(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 1)

    headerotf, raw_psf = read_mrc("./OTF/3D-488-OTF-smallendian.mrc")
    dkr = np.min([dkx, dky])
    nxotf = 101
    nyotf = 257
    nzotf = 3
    dkzotf = 0.06188119
    dkrotf =  0.03201844
    print(headerotf[0])
    print(nxotf * dkzotf)
    z = np.arange(0, nxotf * dkzotf, dkzotf)
    zi = np.arange(0, nxotf * dkzotf, dkz)
    zi = zi[0:-1]
    x = np.arange(0, nyotf * dkrotf, dkrotf)
    xi = np.arange(0, nyotf * dkrotf, dkr)
    xi = xi[0:-1]

    print(raw_psf.shape)
    raw_psf = raw_psf[:, :, 0]
    print(raw_psf.shape)

    OTF1 = []
    for j in range(nxotf):
        curRow = raw_psf[j, :]
        interp = interp1d(x, curRow, 'slinear')
        OTF1.append(interp(xi))
    OTF1 = np.transpose(OTF1, (1, 0))

    OTF2 = []
    for k in range(np.size(OTF1, 0)):
        curCol = OTF1[k]
        interp = interp1d(z, curCol, 'slinear')
        OTF2.append(interp(zi))
    OTF2 = np.transpose(OTF2, (1, 0))

    OTF = F.fftshift(OTF2, 0)
    otflen = np.size(OTF, 1)

    prol_OTF = np.zeros((Nz, diagdist))
    prol_OTF[:, 0: otflen] = OTF
    OTF = prol_OTF

    x = np.arange(-Nx / 2, Nx / 2, 1) * dkx
    y = np.arange(-Ny / 2, Ny / 2, 1) * dky
    [X, Y] = np.meshgrid(x, y)
    rdist = np.sqrt(np.square(X) + np.square(Y))
    curOTF = np.zeros((Ny, Nx, Nz))
    otflen = np.size(OTF, 1)
    x = np.arange(0, otflen * dkr, dkr)
    for z in range(Nz):
        OTFz = OTF[z, :]
        interp = interp1d(x, OTFz, 'slinear')
        curOTF[:, :, z] = interp(rdist)

    curOTF = np.abs(curOTF)
    curOTF = curOTF / np.max(curOTF)
    temp = np.zeros_like(curOTF) + 1j * np.zeros_like(curOTF)
    for j in range(Nz):
        temp[:, :, j] = F.fftshift(F.fft2(np.squeeze(curOTF[:, :, j])))
    PSF = np.abs(F.fftshift(F.fft(temp, axis=2), axes=2))
    PSF = PSF / np.sum(PSF)
    print(PSF.shape)

    # elif 'tif' in otf_path:
    # curOTF = np.transpose(tiff.imread(otf_path),
    #                    (1, 2, 0))
    # PSF = F.fftshift(F.fft(curOTF))
    # curOTF = None
    # PSF = np.abs(PSF)
    # PSF = PSF / np.max(PSF)

    PSF = np.expand_dims(PSF, axis=-1)
    # print(tuple(np.array(dim) - np.array(np.shape(PSF))))
    # quit()

    # for i, idim in enumerate(PSF.shape):
    #     PSF = np.pad(PSF, dim)
    return PSF, curOTF


def gaussian_1d(x, *param):
    """
    :param x:
    :param param:
    :return:
    """
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))


def psf_estimator_3d(psf):
    """
    :param psf:
    :return:
    """
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_y = max_index[0][0]
    index_x = max_index[1][0]
    index_z = max_index[2][0]
    print('in psf estimator')
    print('initial psf shape:', shape)
    print('maximum y, x, z indices:', index_y, index_x, index_z)
    # estimate y sigma
    x = ar(range(shape[0]))
    y = prctile_norm(np.squeeze(psf[:, index_x, index_z]))
    print(x.shape, y.shape)
    fit_y, cov_y = curve_fit(gaussian_1d, x, y, p0=[1, index_y, 2])
    print('estimated psf sigma_y: ', fit_y[2])
    # estimate x sigma
    x = ar(range(shape[1]))
    y = prctile_norm(np.squeeze(psf[index_y, :, index_z]))
    fit_x, cov_x = curve_fit(gaussian_1d, x, y, p0=[1, index_x, 2])
    print('estimated psf sigma_x: ', fit_x[2])
    # estimate z sigma
    x = ar(range(shape[2]))
    y = prctile_norm(np.squeeze(psf[index_y, index_x, :]))
    fit_z, cov_z = curve_fit(gaussian_1d, x, y, p0=[1, index_z, 2])
    print('estimated psf sigma_z: ', fit_z[2])
    return fit_y[2], fit_x[2], fit_z[2]