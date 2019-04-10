import torch.nn as nn
import torch
import numpy as np
import cv2
import torch.nn.functional as functional


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.MSELoss = nn.MSELoss(size_average=False)

    def forward(self, estimated_density_map, gt_map):
        return self.MSELoss(estimated_density_map, gt_map)


class AEBatch(nn.Module):
    def __init__(self):
        super(AEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.abs(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)))


class SEBatch(nn.Module):
    def __init__(self):
        super(SEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.pow(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)), 2)
