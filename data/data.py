from abc import ABC, abstractmethod
import os
import datetime

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

from utils.psf_generator import Parameters3D, cal_psf_3d, psf_estimator_3d

from utils.fcns import prctile_norm, max_norm, min_max_norm, reorder
from utils.fcns import check_folder


class Data(ABC):
    def __init__(self, args):
        self.args = args
        super().__init__()

        norms = {'prctile': prctile_norm,
                 'max': max_norm,
                 'min_max': min_max_norm}
        self.norm = norms[self.args.norm]

        data_name = self.args.dataset

        chkpnt_folder_name = '_'.join([data_name,
                                       self.args.dnn_type,
                                       datetime.datetime.now().strftime("%d-%m-%Y_time%H%M")])

        self.save_weights_path = os.path.join(self.args.checkpoint_dir,
                                              chkpnt_folder_name)

        print(self.save_weights_path)
        check_folder(self.save_weights_path)

        self.sample_path = os.path.join(self.save_weights_path,
                                        'sampled_img')

        self.log_path = os.path.join(self.args.checkpoint_dir,
                                     'graph',
                                     chkpnt_folder_name)
        self.data_dirs = dict()
        self.otf_path = None
        #     check_folder(self.log_path)

        self.input_dim = self.load_sample(self.args.test_dir)
        print('input', self.input_dim)
        self.output_dim = [self.input_dim[0]*2, self.input_dim[1]*2, self.input_dim[2], 1]

    def load_sample(self, path, show=0):
        """
        convert any type of data to [x, y, z, ch] then return dimension
        """
        img = np.transpose(tiff.imread(path), (1, 2, 0))
        img_size = list(np.shape(img))
        if img_size[-1] % (self.args.n_phases * self.args.n_angles) == 0:
            img_size[-1] = img_size[-1] // (self.args.n_phases * self.args.n_angles)
            img_size.append(self.args.n_phases * self.args.n_angles)
        else:
            img_size.append(1)

        if show:
            plt.figure()
            plt.imshow(img[:, :, 0])
            plt.show()

        return img_size

    def data_loader(self,
                    mode, it,
                    batch_size,
                    scale,
                    wf_weight=0):
        """

        Parameters
        ----------
        mode: str
            options: "train" or "test" or "val"
        batch_size
        wf_weight

        Returns
        -------

        todo: plot wf and make sure it is fine
        todo: change percentile norm condition to norm given by user
        """

        images_names = os.listdir(self.data_dirs['x' + mode])
        gt_names = os.listdir(self.data_dirs['y' + mode])

        images_names.sort()
        gt_names.sort()
        x_path = self.data_dirs['x' + mode]
        y_path = self.data_dirs['y' + mode]

        it = it * batch_size
        batch_images_path = images_names[it:batch_size + it]
        gt_images_path = gt_names[it:batch_size + it]

        image_batch = []
        gt_batch = []
        wf_batch = []
        for i in range(len(batch_images_path)):
            cur_img = tiff.imread(os.path.join(x_path,
                                               batch_images_path[i]))
            cur_img[cur_img < 0] = 0

            cur_gt = tiff.imread(os.path.join(y_path,
                                              gt_images_path[i]))
            cur_gt[cur_gt < 0] = 0

            cur_img = self.norm(np.array(cur_img))
            cur_gt = self.norm(np.array(cur_gt))
            image_batch.append(cur_img)
            gt_batch.append(cur_gt)

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)

        nslice = image_batch.shape[1]
        image_batch = np.reshape(image_batch,
                                 (batch_size,
                                  nslice // self.input_dim[2],
                                  self.input_dim[2],
                                  self.input_dim[1],
                                  self.input_dim[0]),
                                 order='F').transpose((0, 3, 4, 2, 1))

        gt_batch = gt_batch.reshape((batch_size,
                                     self.input_dim[2],
                                     self.input_dim[1] * scale,
                                     self.input_dim[0] * scale,
                                     1),
                                    order='F').transpose((0, 2, 3, 1, 4))

        if wf_weight > 0:
            for cur_img in image_batch:
                nchannels = cur_img.shape[-1]

                # WideField is the sum of angles and phases in each z patch
                cur_wf = block_reduce(cur_img,
                                      block_size=(1, 1, 1, nchannels),
                                      func=np.sum,
                                      cval=np.sum(cur_img))
                cur_wf = self.norm(np.array(cur_wf))
                wf_batch.append(cur_wf)

            wf_batch = np.array(wf_batch)
            wf_batch = wf_batch.reshape((batch_size,
                                         self.input_dim[2],
                                         self.input_dim[1],
                                         self.input_dim[0],
                                         1),
                                        order='F').transpose((0, 2, 3, 1, 4))

        return image_batch, gt_batch, wf_batch

    @abstractmethod
    def load_psf(self):
        pass

    def init_psf(self):
        # --------------------------------------------------------------------------------
        #                             Read OTF and PSF
        # --------------------------------------------------------------------------------
        raw_psf = self.load_psf()
        # 128*128*11 otf and psf numpy array
        psf, _ = cal_psf_3d(raw_psf,
                            self.output_dim[:-1])

        print('PSF size before:', np.shape(psf))
        # return psf
        # sigma_y, sigma_x, sigma_z = psf_estimator_3d(psf)  # Find the most effective region of OTF
        ksize = self.output_dim[0] // 2 #int(sigma_y * 4)
        halfy = np.shape(psf)[0] // 2
        print(ksize, halfy)
        psf = psf[halfy - ksize:halfy + ksize, halfy - ksize:halfy + ksize, :]
        print('PSF size after:', np.shape(psf))
        return np.reshape(psf,
                          (ksize * 2, 2 * ksize, np.shape(psf)[2], 1)).astype(np.float32)

