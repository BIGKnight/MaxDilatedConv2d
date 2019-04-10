from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import time
import scipy.io as scio
from utils import GroundTruthProcess, HSI_Calculator
import h5py
import math


class EvalDatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 validate_num,
                 mode
                 ):
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.calcu = HSI_Calculator()
        self.mode = mode
        self.GroundTruthProcess = torch.nn.MaxPool2d(8).cuda()
        for i in range(self.validate_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            height = img.size[1]
            width = img.size[0]
            resize_height = math.ceil(height / 8) * 8
            resize_width = math.ceil(width / 8) * 8
            img = transforms.Resize([resize_height, resize_width])(img)
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])

    def __getitem__(self, index):
        if self.mode == 'whole':
            img, gt_map = self.imgs[index]
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = transforms.ToTensor()(gt_map)
            return index + 1, img.cuda(), gt_map.cuda()

    def __len__(self):
        return self.validate_num
