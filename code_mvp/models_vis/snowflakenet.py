#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
import torch.nn as nn
from .utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer
from .skip_transformer import SkipTransformer
from utils.model_utils import calc_emd, calc_cd, calc_cd_side_sqrt

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr


class SnowflakeNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.up_factors = args.up_factors
        self.train_loss = args.loss
        self.feat_extractor = FeatureExtractor(out_dim=512)
        self.decoder_coarse = SeedGenerator(dim_feat=512, num_pc=256)
        uppers = []
        for i, factor in enumerate(self.up_factors):
            uppers.append(SPD(dim_feat=512, up_factor=factor, i=i, radius=1))

        self.uppers = nn.ModuleList(uppers)

 
    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_1 = self.loss_func(ret[1], gt)
        loss_2 = self.loss_func(ret[2], gt)
        loss_fine = self.loss_func(ret[3], gt)
        return loss_coarse,loss_1, loss_2, loss_fine


    def forward(self, partial, gt, is_training=True, mean_feature=None, alpha=None):
        
        # partial = partial.transpose(2,1).contiguous()
        feat = self.feat_extractor(partial)       
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()
        arr_pcd = []
        arr_pcd.append(pcd)

        pcd = fps_subsample(torch.cat([pcd, partial.transpose(2,1).contiguous()], 1), 512)  # partial B 2048 3  pcd: B 512 3

        K_prev = None
        new_pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            new_pcd, K_prev = upper(new_pcd, feat, K_prev)
            arr_pcd.append(new_pcd.permute(0, 2, 1).contiguous())

        coarse, P1, P2, fine = arr_pcd
        # print(coarse.shape, P1.shape, P2.shape, fine.shape)
        gt_2 = fps_subsample(gt, P2.shape[1])
        gt_1 = fps_subsample(gt, P1.shape[1])
        gt_c = fps_subsample(gt_1, coarse.shape[1])

        if is_training:
            if self.train_loss == 'emd':
                cdc = calc_emd(coarse, gt)
                cd1 = calc_emd(P1, gt)
                cd2 = calc_emd(P2, gt)
                cd3 = calc_emd(fine, gt)
            elif self.train_loss == 'cd':
                # cd_l1做为损失 训练
                cdc, _  = calc_cd(coarse, gt_c)
                cd1, _  = calc_cd(P1, gt_1)
                cd2, _  = calc_cd(P2, gt_2)
                cd3, _  = calc_cd(fine, gt)

            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            loss_all = cdc.mean() + cd1.mean() + cd2.mean() + cd3.mean()
            return fine, cd3, loss_all
        else:
            return fine
            # emd = calc_emd(fine, gt, eps=0.004, iterations=3000)
            # cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)
            # return {'out1': coarse, 'out2': fine, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}

            # return {'out1': coarse, 'out2': fine, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}
