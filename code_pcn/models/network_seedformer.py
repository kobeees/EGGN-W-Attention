import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils_models import MLP_Res,fps_subsample, Attention, SkipTransformer,\
      get_graph_feature, get_nearest_index, PointNetSetAbstraction,\
      PointNetFeaturePropagation, indexing_neighbor, UpTransformer

class SG(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SG, self).__init__()
        self.uptrans = UpTransformer(in_channel=128, out_channel=128, dim=64, n_knn=16, use_upfeat=False, up_factor=2, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch, patch_xyz):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        # print(patch.shape, patch_xyz.shape)
        x1 = self.uptrans(patch_xyz, patch, patch, upfeat=None) # (b, 128, 256)
        # print(x1.shape, feat.shape)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion, x3

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
        partial = fps_subsample( x, self.fps_num).transpose(2,1).contiguous()
        # print(partial.shape)
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

        return global_feature, x3, partial

class idea_with_seedformer(nn.Module):
    def __init__(self, up_factors=[1, 4, 8], fps_num=128):
        super(idea_with_seedformer, self).__init__()

        self.up_factors = up_factors 
        self.fps_num = fps_num
        self.encoder = EAEF(fps_num=self.fps_num)
        self.decoder_coarse = SG(dim_feat=512)
        uppers = []
        for i, factor in enumerate(self.up_factors):
            uppers.append(refine_8(dim_feat=512, up_factor=factor, i=i, radius=1))

        self.uppers = nn.ModuleList(uppers)


    def forward(self, partial):
        feat, patch, patch_xyz = self.encoder(partial)  
        feat = feat.unsqueeze(2)   
        # print(feat.shape, patch_xyz.shape)
        seed, seed_feat = self.decoder_coarse(feat, patch, patch_xyz)
        # print(seed.shape, seed_feat.shape)
        pcd = seed.permute(0, 2, 1).contiguous()
        arr_pcd = []
        arr_pcd.append(pcd)

        pcd = fps_subsample(torch.cat([pcd, partial], 1), 512)  # partial B 2048 3  pcd: B 512 3

        K_prev = None
        new_pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            new_pcd, K_prev = upper(new_pcd, feat, seed, seed_feat, K_prev)
            arr_pcd.append(new_pcd.permute(0, 2, 1).contiguous())
        
        return arr_pcd

class refine_8(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1) -> None:
        super(refine_8, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.interpolate = "three"
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3+3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.2, 32, dim_feat+ 128 +3, [128, 128,256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, dim_feat + 256 +3, [256,256, 512], group_all=True)
    
        self.fp3 = PointNetFeaturePropagation(1792, [256, 256])
        self.fp2 = PointNetFeaturePropagation(1408, [256, 128])
        self.fp1 = PointNetFeaturePropagation(640 , [128, 128])

        self.uptrans1 = UpTransformer(in_channel=128, out_channel=128, dim=64, n_knn=16, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(in_channel=128, out_channel=128, dim=64, n_knn=16, use_upfeat=True, attn_channel=True, up_factor=self.up_factor)
        self.up_sampler = nn.Upsample(scale_factor=up_factor)

        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_delta = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )     

    def forward(self,coarse, feat, seed, seed_feat, K_prev=None):
        b, _, n = coarse.shape
        # Collect seedfeature
        if self.interpolate == 'nearest':
            idx = get_nearest_index(coarse, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3) # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':
            # three interpolate
            idx, dis = get_nearest_index(coarse, seed, k=3, return_dis=True) # (B, N_prev, 3), (B, N_prev, 3)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True) # (B, N_prev, 1)
            weight = dist_recip / norm # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1), dim=-1) # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))
  
        # print(coarse.shape, feat_upsample.shape)
        l1_xyz, l1_points = self.sa1(coarse, coarse)
        l1_points = torch.cat([l1_points, feat.expand(-1,-1, l1_points.shape[2])], dim=1)
        # print("l1_xyz, l1_points", l1_xyz.shape, l1_points.shape)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = torch.cat([l2_points, feat.expand(-1,-1, l2_points.shape[2])], dim=1)
        # print("l2_xyz, l2_points", l2_xyz.shape, l2_points.shape)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.cat([l3_points, feat.expand(-1,-1, l3_points.shape[2])], dim=1)
        # print("l3_xyz, l3_points", l3_xyz.shape, l3_points.shape)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = torch.cat([l2_points, feat.expand(-1,-1, l2_points.shape[2])], dim=1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = torch.cat([l1_points,feat.expand(-1,-1, l1_points.shape[2])], dim=1)

        l0_points = self.fp1(coarse, l1_xyz, None, l1_points)

        H = self.uptrans1(coarse, K_prev if K_prev is not None else l0_points, l0_points, upfeat=feat_upsample)
        feat_child = self.uptrans2(coarse, K_prev if K_prev is not None else H, H, upfeat=feat_upsample)

        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(coarse)

        pcd_child = pcd_child + delta
        
        return pcd_child, K_curr

if __name__ == '__main__':
    input = torch.randn(1, 2048, 3).cuda()
    model = idea_with_seedformer(up_factors=[4,8]).cuda()
    output = model(input)
    for i in range(len(output)):
        print(output[i].shape)
        