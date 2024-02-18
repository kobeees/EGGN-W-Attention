import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils_models import MLP_Res,fps_subsample, Attention, SkipTransformer,\
      get_graph_feature, PointNetSetAbstraction,PointNetFeaturePropagation

class DASG(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(DASG, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.tf_1 = ShiftedChannelMumtiHead(dim_feat + 128, 128, 4)
        self.tf_2 = ShiftedChannelMumtiHead(128,128, 4)
        self.tf_3 = ShiftedChannelMumtiHead(dim_feat + 128 ,128, 4)
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
        x0 = self.ps(feat)  # (b, 128, num_pc)
        x1 = self.tf_1(torch.cat([x0, feat.repeat((1, 1, x0.size(2)))], dim=1))
        x2 = self.tf_2(x1)
        x3 = self.tf_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))
        completion = self.mlp_4(x3)  # (b, 3, 256)
     
        return completion

class ShiftedChannelMumtiHead(nn.Module):
    def __init__(self,in_channels, out_channels=None,psi=4):
        super(ShiftedChannelMumtiHead, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels

        self.to_q = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                  nn.BatchNorm1d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )
        self.to_k = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                  nn.BatchNorm1d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )
        self.to_v = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                  nn.BatchNorm1d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )

        self.w = self.out_channels // psi
        self.d = self.w // 2
        self.heads = 2 * psi - 1

        self.fusion = nn.Sequential(nn.Conv1d(self.heads * self.w,self.out_channels,1),
                                    nn.BatchNorm1d(self.out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self,x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        O = []
        i = 0
        for m in range(self.heads):
            i += 1
            q_m = q[:, m * self.d:m * self.d + self.w,:]
            k_m = k[:, m * self.d:m * self.d + self.w,:]
            v_m = v[:, m * self.d:m * self.d + self.w,:]
            a_m = torch.matmul(q_m.permute(0,2,1),k_m)
            a_m = torch.softmax(a_m,dim=-1)
            o_m = torch.matmul(a_m,v_m.permute(0,2,1))
            O.append(o_m.permute(0,2,1))
        O = torch.cat(O,dim=1)
        return self.fusion(O)

class EAEF(nn.Module):
    def __init__(self, fps_num = 512,k = 16):
        super(EAEF, self).__init__()
        self.k = k
        self.fps_num = fps_num
        self.transformer_1 = Attention(6, pos=3,dim=64)
        self.transformer_2 = Attention(128, pos=3,dim=64)
        self.transformer_3 = Attention(128, pos=3,dim=128)

    def forward(self, x):
        batch_size = x.size(0)
        partial = fps_subsample(x, self.fps_num).transpose(2,1).contiguous()

        x = get_graph_feature(partial, k=self.k)     
        x1 = self.transformer_1(x, x, partial)
        x = get_graph_feature(x1, k=self.k)     
        x2 = self.transformer_2(x, x, partial)
        x = get_graph_feature(x2, k=self.k)    
        x3 = self.transformer_3(x, x, partial)
        x = torch.cat((x1, x2, x3), dim=1)  
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        global_feature = torch.cat((x1, x2), 1)

        return global_feature

# class Refine(nn.Module):
#     def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1) -> None:
#         super(Refine, self).__init__()
#         self.i = i
#         self.up_factor = up_factor
#         self.radius = radius

#         self.FE = LEBP()

#         self.skip_transformer = SkipTransformer(in_channel=128, dim=64)
#         self.up_sampler = nn.Upsample(scale_factor=up_factor)
#         self.mlp_ps = nn.Sequential(
#             nn.Conv1d(128, 64, 1),
#             nn.ReLU(),
#             nn.Conv1d(64, 32, 1)
#         )     
#         self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting
#         self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
#         self.mlp_delta = nn.Sequential(
#             nn.Conv1d(128, 64, 1),
#             nn.ReLU(),
#             nn.Conv1d(64, 3, 1)
#         )     

#     def forward(self,coarse, feat, K_prev=None):
#         b, _, n = coarse.shape

#         l0_points = self.FE(coarse, feat)

#         H = self.skip_transformer(coarse, K_prev if K_prev is not None else l0_points, l0_points)
#         # print("H", H.shape)
#         feat_child = self.mlp_ps(H)
#         # print(f"feat_child_0: ", feat_child.shape)
#         feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
#         # print(f"feat_child: ", feat_child.shape)
#         H_up = self.up_sampler(H)
#         # print(f"H_up: ", H_up.shape)
#         K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

#         delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
#         # print(f"pcd_prev: ", pcd_prev.shape)
#         pcd_child = self.up_sampler(coarse)

#         pcd_child = pcd_child + delta
        
#         return pcd_child, K_curr

# class LEBP(nn.Module):
#     def __init__(self) -> None:
#         super(LEBP, self).__init__()

#         self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3 + 3, [64, 64, 128], False)
#         self.sa2 = PointNetSetAbstraction(128, 0.2, 32, 512 + 128 + 3, [128, 128, 256], False)
#         self.sa3 = PointNetSetAbstraction(None, None, None, 512 + 256 + 3, [256, 256, 512], group_all=True)

#         self.fp3 = PointNetFeaturePropagation(1792, [256, 256])
#         self.fp2 = PointNetFeaturePropagation(1408, [256, 128])
#         self.fp1 = PointNetFeaturePropagation(640, [128, 128])

#     def forward(self, coarse, feat):
#         b, _, n = coarse.shape

#         l1_xyz, l1_points = self.sa1(coarse, coarse)
#         l1_points = torch.cat([l1_points, feat.expand(-1, -1, l1_points.shape[2])], dim=1)
#         # print("l1_xyz, l1_points", l1_xyz.shape, l1_points.shape)

#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l2_points = torch.cat([l2_points, feat.expand(-1, -1, l2_points.shape[2])], dim=1)
#         # print("l2_xyz, l2_points", l2_xyz.shape, l2_points.shape)

#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l3_points = torch.cat([l3_points, feat.expand(-1, -1, l3_points.shape[2])], dim=1)
#         # print("l3_xyz, l3_points", l3_xyz.shape, l3_points.shape)

#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l2_points = torch.cat([l2_points, feat.expand(-1, -1, l2_points.shape[2])], dim=1)
#         # print("l2_points",l2_points.shape)

#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l1_points = torch.cat([l1_points, feat.expand(-1, -1, l1_points.shape[2])], dim=1)
#         # print("l1_points",l1_points.shape)

#         l0_points = self.fp1(coarse, l1_xyz, None, l1_points)

#         return l0_points

class refine_pnplus(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1) -> None:
        super(refine_pnplus, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius

        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3+3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.2, 32, 512+ 128 +3, [128, 128,256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 512 + 256 +3, [256, 256, 512], group_all=True)
    
        self.fp3 = PointNetFeaturePropagation(1792, [256, 256])
        self.fp2 = PointNetFeaturePropagation(1408, [256, 128])
        self.fp1 = PointNetFeaturePropagation(640, [128, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.up_sampler = nn.Upsample(scale_factor=up_factor)

        self.mlp_ps = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1)
        )     
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )     

    def forward(self,coarse, feat, K_prev=None):
        b, _, n = coarse.shape

        l1_xyz, l1_points = self.sa1(coarse, coarse)
        l1_points = torch.cat([l1_points, feat.expand(-1,-1, l1_points.shape[2])], dim=1)
        # print("l1_xyz, l1_points", l1_xyz.shape, l1_points.shape)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print("l2_xyz, l2_points", l2_xyz.shape, l2_points.shape)
        l2_points = torch.cat([l2_points, feat.expand(-1,-1, l2_points.shape[2])], dim=1)
        # print("l2_xyz, l2_points", l2_xyz.shape, l2_points.shape)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print("l3_xyz, l3_points", l3_xyz.shape, l3_points.shape)
        l3_points = torch.cat([l3_points, feat.expand(-1,-1, l3_points.shape[2])], dim=1)
        # print("l3_xyz, l3_points", l3_xyz.shape, l3_points.shape)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = torch.cat([l2_points, feat.expand(-1,-1, l2_points.shape[2])], dim=1)
        # print("l2_points",l2_points.shape)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = torch.cat([l1_points,feat.expand(-1,-1, l1_points.shape[2])], dim=1)
        # print("l1_points",l1_points.shape)

        l0_points = self.fp1(coarse, l1_xyz, None, l1_points)

        H = self.skip_transformer(coarse, K_prev if K_prev is not None else l0_points, l0_points)
        # print("H", H.shape)
        feat_child = self.mlp_ps(H)
        # print(f"feat_child_0: ", feat_child.shape)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        # print(f"feat_child: ", feat_child.shape)
        H_up = self.up_sampler(H)
        # print(f"H_up: ", H_up.shape)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        # print(f"pcd_prev: ", pcd_prev.shape)
        pcd_child = self.up_sampler(coarse)

        pcd_child = pcd_child + delta
        
        return pcd_child, K_curr

class Model(nn.Module):
    def __init__(self, up_factors, fps_num=256):
        super(Model, self).__init__()

        self.up_factors = up_factors
        self.fps_num = fps_num
        self.encoder = EAEF(self.fps_num)
        self.decoder_coarse = DASG(dim_feat=512, num_pc=256)
        uppers = []
        for i, factor in enumerate(self.up_factors):
            uppers.append(refine_pnplus(dim_feat=512, up_factor=factor, i=i, radius=1))
        self.uppers = nn.ModuleList(uppers)

    def forward(self, partial):

        feat = self.encoder(partial)  
        feat = feat.unsqueeze(2)      
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()
        arr_pcd = []
        arr_pcd.append(pcd)

        pcd = fps_subsample(torch.cat([pcd, partial], 1), 512)  # partial B 2048 3  pcd: B 512 3

        K_prev = None
        new_pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            new_pcd, K_prev = upper(new_pcd, feat, K_prev)
            arr_pcd.append(new_pcd.permute(0, 2, 1).contiguous())
        
        return arr_pcd
