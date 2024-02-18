import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import  grouping_operation,\
    gather_operation,furthest_point_sample,ball_query,three_nn, three_interpolate

from knn_cuda import KNN

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def get_graph_feature_(x, k=16, minus_center=True):
    """
    Args:
        x: Tensor of features, (B, in_channel, n)
        k: 

    Returns:
        y: Tensor of features, (B, 2*in_channel, n)  if minus_center==True
    """
    knn = KNN(k = 16, transpose_mode= False)
    dist, idx = knn(x, x)
    idx = idx.transpose(1,2).contiguous() # B npoint k
    # print("idx", idx.shape, idx, idx.dtype)
    feature = grouping_operation(x , idx.int()).permute(0, 1, 3, 2).cuda() # b, dim, k, n
    # print("feature", feature.shape)
    x = x.unsqueeze(2).repeat(1,1,k,1)

    if minus_center:
        feature = torch.cat((x, feature - x), dim=1).permute(0, 1, 3, 2).contiguous()
    else:
        feature = torch.cat((x, feature),dim=1).permute(0, 1, 3, 2).contiguous()
    return feature

def knn_dg(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    # print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=16, minus_center=True):
    """
    Args:
        x: Tensor of features, (B, in_channel, n)
        k: 

    Returns:
        y: Tensor of features, (B, 2*in_channel, n)  if minus_center==True
    """
    idx = knn_dg(x, k)
    # print("idx: ", idx.shape)
    batch_size, num_points, _ = idx.size()
    device = idx.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # print("feature", feature.shape )
    feature = feature.view(batch_size, num_points, k, num_dims)
    print("feature", feature.shape )
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    print("feature", feature.shape )
    if minus_center:
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    else:
        feature = torch.cat((x, feature), dim=3).permute(0, 3, 1, 2)
    return feature

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # print("src: ",src.shape, dst.shape)
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample+pad]
    return idx.int()

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out

class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU()

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)

class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        # print("value: ", value.shape)
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        # print("query: ", query.shape)
        value = self.conv_value(value)
        b, dim, n = value.shape
        # print("value: ", value.shape)
        # idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)
        knn = KNN(k = self.n_knn, transpose_mode= False)
        dist, idx = knn(pos, pos)
        idx_knn = idx.transpose(1,2).contiguous().int() # B npoint k
        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #
        # print("value: ", value.shape)
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)
        # print("y: ", y.shape)
        return y + identity

class Attention(nn.Module):
    def __init__(self, in_channel, pos=3,dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Attention, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv2d(in_channel, dim, 1)
        self.conv_query = nn.Conv2d(pos, dim, 1)
        self.conv_value = nn.Conv2d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(dim, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_end = nn.Conv1d(dim, dim, 1)

    def forward(self, key, values, pos):
        """feed forward of transformer
        Args:
            key: Tensor of features, (B, in_channel, n, k)
            values: Tensor of features, (B, in_channel, n, k)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        # identity = x
        b, dim, n, _ = key.shape

        knn = KNN(k = self.n_knn, transpose_mode= False)
        dist, idx = knn(pos, pos)
        idx_knn = idx.transpose(1,2).contiguous().int() # B npoint k
        pos_flipped = grouping_operation(pos, idx_knn)  # b, dim, n, n_knn 

        key = self.conv_key(key) # B dim n k 
        value = self.conv_value(values)
        query = self.conv_query(pos_flipped)

        qk_rel = query - key

        pos_embedding = self.pos_mlp(query)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel * pos_embedding + pos_embedding)
        attention = torch.softmax(attention, -1)
        value = value + pos_embedding

        # einsum: 爱因斯坦求和约定（einsum）提供了一套既简洁又优雅的规则，
        # 可实现包括但不限于：向量内积，向量外积，
        # 矩阵乘法，转置和张量收缩（tensor contraction）等张量操作，
        # 熟练运用 einsum 可以很方便的实现复杂的张量操作，而且不容易出错。
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y
   
class Transformer_new(nn.Module):
    def __init__(self, in_channel, pos=3,dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer_new, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv2d(in_channel, dim, 1)
        self.conv_query = nn.Conv2d(pos, dim, 1)
        self.conv_value = nn.Conv2d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(dim, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_end = nn.Conv1d(dim, dim, 1)

    def forward(self, key, values, pos):
        """feed forward of transformer
        Args:
            key: Tensor of features, (B, in_channel, n, k)
            values: Tensor of features, (B, in_channel, n, k)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        # identity = x
        b, dim, n, _ = key.shape

        pos_flipped = pos.unsqueeze(3).expand(b, -1, -1, self.n_knn) 
        key = self.conv_key(key) # B dim n k 
        value = self.conv_value(values)
        query = self.conv_query(pos_flipped)

        qk_rel = query - key

        pos_embedding = self.pos_mlp(query)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel * pos_embedding + pos_embedding)
        attention = torch.softmax(attention, -1)
        value = value + pos_embedding

        # einsum: 爱因斯坦求和约定（einsum）提供了一套既简洁又优雅的规则，
        # 可实现包括但不限于：向量内积，向量外积，
        # 矩阵乘法，转置和张量收缩（tensor contraction）等张量操作，
        # 熟练运用 einsum 可以很方便的实现复杂的张量操作，而且不容易出错。
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y
  
class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius
    parameters:
        radius: float32, Radius of ball
        nsample: int32, Maximum number of features to gather in the ball
    """
    def __init__(self, radius=None, nsample=32, use_xyz=True, return_idx=False):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample  = radius, nsample
        self.use_xyz = use_xyz
        self.return_idx = return_idx

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor = None, features: torch.Tensor = None, idx: torch.Tensor = None) -> torch.Tensor:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
               features: (b, c, n)
               idx: idx of neighbors
               # idxs: (b, n)
        output: new_features: (b, c+3, m, k)
              #  grouped_idxs: (b, m, k)
        """
        if new_xyz is None:
            new_xyz = xyz.contiguous()
            # print("new_xyz",new_xyz.shape)
        if idx is None:
            if self.radius is not None:
                idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            else:
                knnn = KNN(k = self.nsample, transpose_mode= False)
                dist, idx = knnn(xyz.contiguous(), new_xyz)
                idx = idx.transpose(1,2).contiguous()
                # idx = knn(self.nsample, xyz.contiguous(), new_xyz)  # (b, m, k)
        grouped_xyz = grouping_operation(xyz.transpose(1, 2).contiguous(), idx)  # (b, 3, m, k)
        grouped_xyz_diff = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz_diff, grouped_features], dim=1)  # (b, 3+c, m, k)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz_diff

        if self.return_idx:
            return new_features, grouped_xyz, idx.long()
            # (b,c,m,k), (b,3,m,k), (b,m,k)
        else:
            return new_features, grouped_xyz

class QueryAndGroupFeature(nn.Module):
    """
    Groups with a ball query of radius
    parameters:
        radius: float32, Radius of ball
        nsample: int32, Maximum number of features to gather in the ball
    """
    def __init__(self, radius=None, nsample=32, use_feature=True, return_idx=False):
        super(QueryAndGroupFeature, self).__init__()
        self.radius, self.nsample, self.use_feature = radius, nsample, use_feature
        self.return_idx = return_idx

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor = None, features: torch.Tensor = None, idx: torch.Tensor = None) -> torch.Tensor:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
               features: (b, c, n)
               idx: idx of neighbors
               # idxs: (b, n)
        output: new_features: (b, c+3, m, nsample)
              #  grouped_idxs: (b, m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        if idx is None:
            if self.radius is not None:
                idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            else:
                knnn = KNN(k = self.nsample, transpose_mode= False)
                dist, idx = knnn(xyz, new_xyz)
                idx = idx.transpose(1,2).contiguous() # B npoint k
                # idx = knn(self.nsample, xyz, new_xyz)  # (b, m, nsample)
        grouped_features = grouping_operation(features, idx)
        grouped_featuresd_diff = grouped_features - features.unsqueeze(-1)
        if self.use_feature:
            new_features = torch.cat([grouped_featuresd_diff, grouped_features], dim=1)  # (b, 3+c, m, nsample)
        else:
            new_features = grouped_features

        if self.return_idx:
            return new_features, idx.long()
            # (b,c,m,k), (b,3,m,k), (b,m,k)
        else:
            return new_features

def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()) # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz

def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous() # (B, N, 3)
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint)) # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx) # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx) # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz
