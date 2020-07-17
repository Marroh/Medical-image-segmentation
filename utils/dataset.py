from os.path import splitext, exists, join
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms as T
from random import randint
import SimpleITK as sitk
from cv2 import *

transform = T.Compose([T.ToTensor(),
                      T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        # self.scale = scale
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod

    def preprocess(cls, img, mask, randomcrop_size):
        mask = np.array(mask)
        img = transform(img)
        assert randomcrop_size[0] < mask.shape[0] and randomcrop_size[1] < mask.shape[1], print('Crop size is too large',randomcrop_size,img.shape,mask.shape)
        assert img.shape[1:] == mask.shape, print('img and mask doesnt match', img.shape, mask.shape)
        x0 = randint(0, mask.shape[0]-randomcrop_size[0])
        y0 = randint(0, mask.shape[1]-randomcrop_size[1])
        # img_crop = np.zeros(randomcrop_size.append(img.shape[-1]))
        # mask_crop = np.zeros(randomcrop_size)
        img_crop = img[:, x0:x0+randomcrop_size[0], y0:y0+randomcrop_size[1]]
        # img_crop = np.expand_dims(img_crop, axis=1)
        mask_crop = mask[x0:x0+randomcrop_size[0], y0:y0+randomcrop_size[1]]

        # print(np.array(img_crop).shape,mask_crop.shape)

        return np.array(img_crop), np.array(mask_crop)

    # def preprocess(cls, img, scale, mask=False):
    #     #TODO 输入img归一化 *update：已在__getitem__中归一化
    #     img = transform(img)
    #     # img.show()
    #     img_nd = np.array(img)
    #     # if len(img_nd.shape) == 2:
    #     #     img_nd = np.expand_dims(img_nd, axis=2)
    #     #
    #     # # HWC to CHW
    #     # img_trans = img_nd.transpose((2, 0, 1))
    #     # if img_nd.max() > 1:
    #     #     img_nd = img_nd / 255
    #
    #     return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        npmask = np.loadtxt(mask_file[0])+1
        # npmask[npmask<5]=255.0
        mask = Image.fromarray(npmask)#cause unkown objects were labled -1
        img = Image.open(img_file[0])
        # print('\nbefore transform', (np.loadtxt(mask_file[0])+1).min(), (np.loadtxt(mask_file[0])+1).max())
        # print('img size: ', img.shape)
        # print('mask size: ', mask.shape)

        #assert img.size == mask.size, \
        #   f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.preprocess(img, mask, [120,180])
        print('\nelements in mask', np.unique(mask))
        #print('after transform', (torch.from_numpy(img)).min(), (torch.from_numpy(img)).max())
        # print(torch.from_numpy(img).shape,torch.from_numpy(mask).shape)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}

class DCM_Dataset(Dataset):
    def __init__(self, imgs_dir):
        structures = ['BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R',
                      'Submandibular_L', 'Submandibular_R']
        self.imgs_dir = imgs_dir
        # self.scale = scale
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        #print(listdir(imgs_dir))
        self.ids = [file for file in sorted(listdir(imgs_dir))
                    if 'image' in file and 'mhd' in file]
        #print(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    #@classmethod
    # def normalize(self, matrix_):
    #     minval = matrix_.min()
    #     tmpImage = (matrix_ - minval).astype('float')
    #     if tmpImage.max() != 0:
    #         rescaledImage = ((tmpImage / tmpImage.max()) * 255.0).astype(np.float32)  # 归一化
    #     else:
    #         rescaledImage = tmpImage.astype(np.float32)
    #
    #     #print(rescaledImage.dtype)
    #     return rescaledImage

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_path = idx.split('_')[0] + '_label-'+idx.split('_')[-1].split('-')[1]+'-t.mhd'

        img_file = join(self.imgs_dir, idx)
        mask_file = join(self.imgs_dir, str(mask_path))

        npmask = sitk.ReadImage(mask_file)
        npmask = sitk.GetArrayFromImage(npmask)

        npimg = sitk.ReadImage(img_file)
        npimg = sitk.GetArrayFromImage(npimg)

        # print(npimg.shape, npmask.shape)
        # imshow("img", ((npimg[1]+200)/400*255).astype(np.uint8))
        # imshow("mask", npmask)
        # waitKey(30)

        return {'image': torch.from_numpy(npimg), 'mask': torch.from_numpy(npmask)}

# data = DCM_Dataset('E:/Data_set/SAMPLE')
# for i in range(100):
#     data.__getitem__(i)