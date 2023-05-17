"""
Implementation of Where2comm fusion.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention
import os
import shutil

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """

        _, _, H, W = batch_confidence_maps[0].shape
        communication_masks = []
        communication_rates = []
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            L = communication_maps.shape[0]
            if self.training:
                # Official training proxy objective
                K = int(H * W * random.uniform(0, 1))
                communication_maps = communication_maps.reshape(L, H * W)
                _, indices = torch.topk(communication_maps, k=K, sorted=False)
                communication_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                ones_fill = torch.ones(L, K, dtype=communication_maps.dtype, device=communication_maps.device)
                communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(L, 1, H, W)
            elif self.threshold:
                ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
                zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
                communication_mask = torch.where(communication_maps > self.threshold, ones_mask, zeros_mask)
            else:
                communication_mask = torch.ones_like(communication_maps).to(communication_maps.device)

            communication_rate = communication_mask.sum() / (L * H * W)
            # Ego
            communication_mask[0] = 1

            communication_masks.append(communication_mask)
            communication_rates.append(communication_rate)
        communication_rates = sum(communication_rates) / B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates


class ShortTermAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ShortTermAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class EnhanceWeight(nn.Module):
    def __init__(self):
        super(EnhanceWeight, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.tanhAug = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanhAug(x) + torch.Tensor([1.0]).cuda()
        return x

class EnhanceWeightConfm(nn.Module):
    def __init__(self):
        super(EnhanceWeightConfm, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.tanhAug = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanhAug(x) + torch.Tensor([1.0]).cuda()
        return x

class TransformerFusion(nn.Module):
    def __init__(self, feature_dim):
        super(TransformerFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class HPHA(nn.Module):
    def __init__(self, args):
        super(HPHA, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']
        if self.fully:
            print('constructing a fully connected communication graph')
        else:
            print('constructing a partially connected communication graph')

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            # layer_nums,self.num_levels,num_filters: [3, 5, 8] 3 [64, 128, 256]
            for idx in range(self.num_levels):
                fuse_network = TransformerFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = TransformerFusion(args['in_channels'])

        self.naive_communication = Communication(args['communication'])
        self.sta = ShortTermAttention(512)
        self.enhanceweight = EnhanceWeight()
        self.enhanceweight_confm = EnhanceWeightConfm()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    def forward(self, x, historical_x, psm_single, record_len, pairwise_t_matrix, time_delay, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        _, C, H, W = x.shape
        B = pairwise_t_matrix.shape[0]

        ## semantic information enhance based on IoSI
        x_enw = torch.zeros_like(x) ## semantic information enhance weight
        historical_x_enw = torch.zeros_like(historical_x) ## historical semantic information enhance weight
        aoi_time_delay = time_delay[0]
        for t in range(x.shape[0] + historical_x.shape[0]):
            if t == 1 or t == 2:
                historical_x_enw[t - 1] = self.enhanceweight(torch.tensor([1/(aoi_time_delay[t]+0.1)], dtype=torch.float32).cuda())
            elif t >= 3:
                x_enw[t - 2] = self.enhanceweight(torch.tensor([1/(aoi_time_delay[t]+0.1)], dtype=torch.float32).cuda())
            else:
                x_enw[t] = self.enhanceweight(torch.tensor([1/(aoi_time_delay[t]+0.1)], dtype=torch.float32).cuda())
        x = x * x_enw
        historical_x = historical_x * historical_x_enw

        historical_x = backbone.blocks[0](historical_x)
        ## semantic information Aggregated based on multi-scale transformer module
        if self.multi_scale:
            ups = []
            for i in range(self.num_levels):
                x = backbone.blocks[i](x)
                # 1. Communication (mask the features)
                if i == 0:
                    if self.fully:
                        communication_rates = torch.tensor(1).to(x.device)
                    else:
                        # Prune
                        batch_confidence_maps = self.regroup(psm_single, record_len)  ## batch_confidence_maps.shape -> [1, 4, 2, 48, 176],B=1
                        ## semantic information enhance based on IoSI
                        for k in range(batch_confidence_maps[0].shape[0]):
                            if k == 0:
                                batch_confidence_maps[0][k] = batch_confidence_maps[0][k] * self.enhanceweight_confm(torch.tensor([1/(aoi_time_delay[k]+0.1)],dtype=torch.float32).cuda())
                            else:
                                batch_confidence_maps[0][k] = batch_confidence_maps[0][k] * self.enhanceweight_confm(torch.tensor([1/(aoi_time_delay[k+2]+0.1)],dtype=torch.float32).cuda())
                        # ##
                        communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                        if x.shape[-1] != communication_masks.shape[-1]:  ##
                            communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                                mode='bilinear', align_corners=False)
                        x = x * communication_masks


                # 2. Split the features
                # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
                # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
                batch_node_features = self.regroup(x, record_len)

                # 3. Fusion
                x_fuse = []
                for b in range(B):
                    neighbor_feature = batch_node_features[b]
                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)
                # 4. Deconv
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)
            ups.append(historical_x[0].unsqueeze(0))
            ups.append(historical_x[1].unsqueeze(0))
            if len(ups) > 1:
                x_fuse = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x_fuse = ups[0]
            ## semantic information refined based on short-term attention module
            x_fuse = self.sta(x_fuse) * x_fuse
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:
            # 1. Communication (mask the features)
            if self.fully:
                communication_rates = torch.tensor(1).to(x.device)
            else:
                # Prune
                batch_confidence_maps = self.regroup(psm_single, record_len)
                communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, B)
                x = x * communication_masks

            # 2. Split the features
            # split_x: [(L1, C, H, W), (L2, C, H, W), ...]
            # For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        return x_fuse, communication_rates
