import torch
import torch.nn as nn
from torch import einsum

from utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import  grouping_operation,\
    gather_operation,furthest_point_sample,ball_query,three_nn, three_interpolate
from utils.model_utils import calc_emd, calc_cd

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

class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

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

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape
        # print("pos", pos.shape)

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y+identity


class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


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


class PointNet_SA_Module(nn.Module):
    def __init__(self, npoint, nsample, radius, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            radius: float
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(xyz, points, self.npoint, self.nsample, self.radius, self.use_xyz)

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        return new_xyz, new_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None, if_bn=True):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, in_channel, N)
            points2: Tensor, (B, in_channel, M)

        Returns:
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0/dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight) # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points



class Unit(nn.Module):
    def __init__(self, step=1, in_channel=256):
        super(Unit, self).__init__()
        self.step = step
        if step == 1:
            return

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.relu)

    def forward(self, cur_x, prev_s):
        """
        Args:
            cur_x: Tensor, (B, in_channel, N)
            prev_s: Tensor, (B, in_channel, N)

        Returns:
            h: Tensor, (B, in_channel, N)
            h: Tensor, (B, in_channel, N)
        """
        if self.step == 1:
            return cur_x, cur_x

        z = self.conv_z(torch.cat([cur_x, prev_s], 1))
        r = self.conv_r(torch.cat([cur_x, prev_s], 1))
        h_hat = self.conv_h(torch.cat([cur_x, r * prev_s], 1))
        h = (1 - z) * cur_x + z * h_hat
        return h, h


class StepModel(nn.Module):
    def __init__(self, step=1):
        super(StepModel, self).__init__()
        self.step = step
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3, [64, 64, 128], group_all=False)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        # print('l1_xyz, l1_points', l1_xyz.shape, l1_points.shape)
        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        # print('l2_xyz, l2_points', l2_xyz.shape, l2_points.shape)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)
        # print('l3_xyz, l3_points', l3_xyz.shape, l3_points.shape)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])
        # print('l2_points, prev_s[l2]', l2_points.shape, prev_s['l2'].shape)

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])
        # print('l1_points, prev_s[l1]', l1_points.shape, prev_s['l1'].shape)

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)
        # print('l0_points, prev_s[l0]', l0_points.shape, prev_s['l0'].shape)

        b, _, n = l0_points.shape
        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz


class StepModelNoise(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2):
        super(StepModelNoise, self).__init__()
        self.step = step
        self.if_noise = if_noise
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3 + (self.noise_dim if self.if_noise else 0), [64, 64, 128],
                                              group_all=False)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True,
                                              in_channel_points1=6 + (self.noise_dim if self.if_noise else 0))

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)

        mlp = [128, 64, 3]
        last_channel = 128 + 32
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud

        b, _, n = l0_points.shape

        noise_points = torch.normal(mean=0, std=torch.ones((b, (self.noise_dim if self.if_noise else 0), n),
                                                           device=device) * self.noise_stdv)
        l0_points = torch.cat([l0_points, noise_points], 1)

        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)

        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)

        noise = torch.normal(mean=0, std=torch.ones((b, 32, n), device=device))
        delta_xyz = torch.tanh(self.mlp_conv(torch.cat([l0_points, noise], 1))) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz


class PMPNet(nn.Module):
    def __init__(self, args, dataset='Completion3D', noise_dim=3, noise_stdv=1e-2):
        super(PMPNet, self).__init__()
        if dataset == 'ShapeNet':
            self.step_1 = StepModelNoise(step=1, if_noise=True, noise_dim=noise_dim, noise_stdv=noise_stdv)
        else:
            self.step_1 = StepModel(step=1)

        self.step_2 = StepModelNoise(step=2)
        self.step_3 = StepModelNoise(step=3)
        self.train_loss = args.loss

    def forward(self, point_cloud, gt, is_training=True, mean_feature=None, alpha=None):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, _, npoint = point_cloud.shape
        device = point_cloud.device
        # point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        prev_s = {
            'l0': torch.normal(mean=0, std=torch.ones((b, 128, npoint), dtype=torch.float, device=device) * 0.01),
            'l1': torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device) * 0.01),
            'l2': torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device) * 0.01)
        }

        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        pcds = [pcd_out_1.permute(0, 2, 1).contiguous(), pcd_out_2.permute(0, 2, 1).contiguous(),
                pcd_out_3.permute(0, 2, 1).contiguous()]

        deltas = [delta1, delta2, delta3]
        if is_training:
            if self.train_loss == 'emd':
                pass
            elif self.train_loss == 'cd':
                cd1, _ = calc_cd(pcds[0], gt)
                cd2, _ = calc_cd(pcds[1], gt)
                cd3, _ = calc_cd(pcds[2], gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            delta_losses = []
            for delta in deltas:
                delta_losses.append(torch.sum(delta ** 2))
            loss_pmd = torch.sum(torch.stack(delta_losses)) / 3

            cd_loss = cd1.mean() + cd2.mean() + cd3.mean()

            total_train_loss = cd_loss * 1000  + loss_pmd * 1e-2
            return pcds[2], cd3, total_train_loss
        else:
            # cd_p, cd_t, f1 = calc_cd(pcds[2], gt, calc_f1=True)
            return pcds[2]
            # return {'out1': out1, 'out2': out2, 'emd': 0, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}


class StepModelTransformer(nn.Module):
    def __init__(self, step=1, if_noise=False, noise_dim=3, noise_stdv=1e-2, dim_tail=32):
        super(StepModelTransformer, self).__init__()
        self.step = step
        self.if_noise = if_noise
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.dim_tail = dim_tail
        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3 + (self.noise_dim if self.if_noise else 0), [64, 64, 128],
                                              group_all=False)
        self.transformer_start_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.transformer_start_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 1024], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(1024, [256, 256], use_points1=True, in_channel_points1=256)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        self.unit_3 = Unit(step=step, in_channel=256)
        self.unit_2 = Unit(step=step, in_channel=128)
        self.unit_1 = Unit(step=step, in_channel=128)


        mlp = [128, 64, 3]
        last_channel = 128 + self.dim_tail  # (32 if self.step == 1 else 0)
        mlp_conv = []
        for out_channel in mlp[:-1]:
            mlp_conv.append(Conv1d(last_channel, out_channel, if_bn=True))
            last_channel = out_channel
        mlp_conv.append(Conv1d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*mlp_conv)

    def forward(self, point_cloud, prev_s):
        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud
        b, _, n = l0_points.shape
        noise_points = torch.normal(mean=0, std=torch.ones((b, (self.noise_dim if self.if_noise else 0), n),
                                                           device=device) * self.noise_stdv)
        l0_points = torch.cat([l0_points, noise_points], 1)
        l1_xyz, l1_points = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_start_1(l1_points, l1_xyz)

        l2_xyz, l2_points = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_start_2(l2_points, l2_xyz)

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, 1024, 1)

        l2_points = self.fp_module_3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points, prev_s['l2'] = self.unit_3(l2_points, prev_s['l2'])

        l1_points = self.fp_module_2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points, prev_s['l1'] = self.unit_2(l1_points, prev_s['l1'])

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_xyz], 1), l1_points)
        l0_points, prev_s['l0'] = self.unit_1(l0_points, prev_s['l0'])  # (B, 128, 2048)

        noise = torch.normal(mean=0, std=torch.ones((b, self.dim_tail, n), device=device) * self.noise_stdv)
        l0_points = torch.cat([l0_points, noise], 1)
        delta_xyz = torch.tanh(self.mlp_conv(l0_points)) * 1.0 / 10 ** (self.step - 1)
        point_cloud = point_cloud + delta_xyz
        return point_cloud, delta_xyz

class PMPNetPlus(nn.Module):
    def __init__(self,args, dataset='Completion3D', dim_tail=32):
        super(PMPNetPlus, self).__init__()
        self.step_1 = StepModelTransformer(step=1, if_noise=True, dim_tail=dim_tail)
        self.step_2 = StepModelTransformer(step=2, if_noise=True, dim_tail=dim_tail)
        self.step_3 = StepModelTransformer(step=3, if_noise=True, dim_tail=dim_tail)
        # self.step_4 = StepModelNoiseTransformerPCN(step=4, if_noise=True, dim_tail=dim_tail)
        self.train_loss = args.loss

    def forward(self, point_cloud, gt, is_training=True, mean_feature=None, alpha=None):
        """
        Args:
            point_cloud: Tensor, (B, 2048, 3)
        """
        b, _, npoint = point_cloud.shape
        device = point_cloud.device
        # point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        prev_s = {
            'l0': torch.normal(mean=0, std=torch.ones((b, 128, npoint), dtype=torch.float, device=device) * 0.01),
            'l1': torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device) * 0.01),
            'l2': torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device) * 0.01)
        }

        pcd_out_1, delta1 = self.step_1(point_cloud, prev_s)
        pcd_out_2, delta2 = self.step_2(pcd_out_1, prev_s)
        pcd_out_3, delta3 = self.step_3(pcd_out_2, prev_s)

        pcds = [pcd_out_1.permute(0, 2, 1).contiguous(), pcd_out_2.permute(0, 2, 1).contiguous(),
                pcd_out_3.permute(0, 2, 1).contiguous()]

        deltas = [delta1, delta2, delta3]
        if is_training:
            if self.train_loss == 'emd':
                pass
            elif self.train_loss == 'cd':
                cd1, _ = calc_cd(pcds[0], gt)
                cd2, _ = calc_cd(pcds[1], gt)
                cd3, _ = calc_cd(pcds[2], gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            delta_losses = []
            for delta in deltas:
                delta_losses.append(torch.sum(delta ** 2))
            loss_pmd = torch.sum(torch.stack(delta_losses)) / 3

            cd_loss = cd1.mean() + cd2.mean() + cd3.mean()

            total_train_loss = cd_loss * 1000 + loss_pmd * 1e-2
            return pcds[2], cd3, total_train_loss
        else:
            return pcds[2]