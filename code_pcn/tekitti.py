import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from models.network import Model
# from models.PoinTr import PoinTr
# from models.GRNet import GRNet
# from models.Snowflake import SnowflakeNet
# from models.PCN import PCN

from dataset.kitti import KITTI
from dataset.shapenet import ShapeNetPcd
# from utils.misc import visualize_KITTI
from utils.visualization import plot_pcd_one_view
from utils.utils import make_dir, get_logger, create_dir
from extensions.chamfer_dist import ChamferDistanceL2_split, ChamferDistanceL2

def test(params):
    # # load pretrained model PoinTr
    # model = PoinTr().cuda()
    # path_checkpoint = f"./{ckpt_dir}/KITTI.pth"
    # state_dict = torch.load(path_checkpoint, map_location='cpu')
    # # parameter resume of base model
    # if state_dict.get('model') is not None:
    #     base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    # elif state_dict.get('base_model') is not None:
    #     base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    # else:
    #     raise RuntimeError('mismatch of ckpt weight')
    # model.load_state_dict(base_ckpt)

    # # load pretrained model GRNet
    # model = GRNet().cuda()
    # print(model)
    # path_checkpoint = f"./{ckpt_dir}/GRNet-KITTI.pth"
    # checkpoint = torch.load(path_checkpoint)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['grnet'].items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    # # load pretrained model  PCN
    # model = PCN().cuda()
    # path_checkpoint = f"./explijl/PCN/KITTI_lr_0.001_metric_mmd/checkpoints/best_l2_cd.pth"
    # checkpoint = torch.load(path_checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # load our model
    model = Model(dim_feat=512, up_factors=params.up_factors).cuda()
    # model = new_fast(up_factors = params.up_factors).cuda()
    path_checkpoint = f"./{ckpt_dir}/ckpt-best-KITTI.pth"
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loading weights from {path_checkpoint}")

    # model = nn.DataParallel(model)

    test_dataset = KITTI(dataroot='./data/kitti', split='test')
    testDataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    # print(len(test_dataset))
    with tqdm(testDataloader) as t:
        for idx, (taxonomy_ids, model_ids, data) in enumerate(t):
            model_id = model_ids[0]
            partial = data.cuda()
            ret = model(partial)
            dense_points = ret[-1]
            target_path = result_dir
            if not os.path.exists(target_path):
                os.mkdir(target_path)

            plot_pcd_one_view(os.path.join(target_path, '{:03d}_sum.png'.format(idx)),
                              [partial[0].detach().cpu().numpy(), dense_points[0].detach().cpu().numpy()],
                              ['Partial', 'Ours'],
                              xlim=(-0.7, 0.7), ylim=(-0.7, 0.7), zlim=(-0.7, 0.7))


def build_ShapeNetCars(params):
    train_dataset = ShapeNetPcd('data/ShapeNet', split='train', category=params.category)
    val_dataset = ShapeNetPcd('data/ShapeNet', split='val', category=params.category)
    test_dataset = ShapeNetPcd('data/ShapeNet', split='test', category=params.category)
    CarsDataset = train_dataset + val_dataset + test_dataset
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return CarsDataset


def get_Fidelity():
    # Fidelity Error
    criterion = ChamferDistanceL2_split(ignore_zeros=True)
    # Your data
    Samples = [item for item in os.listdir(result_dir) if os.path.isdir(result_dir + '/' + item)]
    metric = []
    for sample in Samples:
        input_data = torch.from_numpy(np.load(os.path.join(result_dir, sample, 'input.npy'))).unsqueeze(0).cuda()
        pred_data = torch.from_numpy(np.load(os.path.join(result_dir, sample, 'pred.npy'))).unsqueeze(0).cuda()
        metric.append(criterion(input_data, pred_data)[0])
    logger.info('Fidelity is %f' % (sum(metric) / len(metric)))


def get_Consistency():
    # Consistency
    criterion = ChamferDistanceL2(ignore_zeros=True)
    Cars_dict = {}
    for sample in Samples:
        all_elements = sample.split('_')  # example sample = 'frame_1_car_3_647'
        frame_id = int(all_elements[1])
        car_id = int(all_elements[-2])
        sample_id = int(all_elements[-1])

        if Cars_dict.get(car_id) is None:
            Cars_dict[car_id] = [f'frame_{frame_id:03d}_car_{car_id:02d}_{sample_id:03d}']
        else:
            Cars_dict[car_id].append(
                f'frame_{frame_id:03d}_car_{car_id:02d}_{sample_id:03d}')  # example sample = 'frame_001_car_003_647'

    Consistency = []
    for key, car_list in Cars_dict.items():
        car_list = sorted(car_list)
        Each_Car_Consistency = []
        for i, this_car in enumerate(car_list):
            if i == len(car_list) - 1:
                break
            this_elements = this_car.split('_')
            this_frame = int(this_elements[1])

            next_car = car_list[i + 1]
            next_elements = next_car.split('_')
            next_frame = int(next_elements[1])

            if next_frame - 1 != this_frame:
                continue

            this_car = torch.from_numpy(np.load(
                os.path.join(result_dir, f'frame_{this_frame}_car_{int(this_elements[3])}_{int(this_elements[4]):03d}',
                             'pred.npy'))).unsqueeze(0).cuda()
            next_car = torch.from_numpy(np.load(
                os.path.join(result_dir, f'frame_{next_frame}_car_{int(next_elements[3])}_{int(next_elements[4]):03d}',
                             'pred.npy'))).unsqueeze(0).cuda()
            cd = criterion(this_car, next_car)
            Each_Car_Consistency.append(cd)

        MeanCD = sum(Each_Car_Consistency) / len(Each_Car_Consistency)
        Consistency.append(MeanCD)
    MeanCD = sum(Consistency) / len(Consistency)
    print(f'Consistency is {MeanCD:.6f}')


def get_MMD():
    criterion = ChamferDistanceL2(ignore_zeros=True)
    Samples = [item for item in os.listdir(result_dir) if os.path.isdir(result_dir + '/' + item)]
    # MMD
    metric = []
    for item in tqdm(sorted(Samples)):
        print(item)
        pred_data = torch.from_numpy(np.load(os.path.join(result_dir, item, 'pred.npy'))).unsqueeze(0).cuda()
        batch_cd = []
        # for index in range(len(ShapeNetCars_dataset)):
        # print("ShapeNetCars_dataset[index]:", ShapeNetCars_dataset[index].shape) # 'tuple' object has no attribute 'shape'
        # print("ShapeNetCars_dataset[index][-1]", ShapeNetCars_dataset[index][-1].shape) # 16384 3
        # gt = ShapeNetCars_dataset[index][-1].cuda().unsqueeze(0) # 1 16384 3
        for index, (partial, gt) in enumerate(CarsDataloader):
            gt = gt.cuda()
            batch_pred_data = pred_data.expand(gt.size(0), -1, -1).contiguous()
            min_cd = criterion(gt, pred_data)
            batch_cd.append(min_cd)
        min_cd = min(batch_cd).item()
        metric.append(min_cd)
        logger.info('This item %s CD %f, MMD %f' % (item, min_cd, sum(metric) * 1.0 / len(metric)))
    logger.info('MMD is %f' % (sum(metric) / len(metric)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--vis_path', type = str,  help = 'KITTI visualize path')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Tag of experiment')
    parser.add_argument('--datasets', type=str, default='KITTI', help='Datasets')
    parser.add_argument('--model', type=str, default='idea_with_fps_256', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, help='The path of pretrained model.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--metric', type=str, default='cd_l1', help='mertic for Testing')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')

    # model
    parser.add_argument('--up_factors', default=[1, 4, 8], help='for upsampling')
    parser.add_argument('--fps_num', type=int, default=256, help='fps_num')

    args = parser.parse_args()
    log_dir, ckpt_dir, epochs_dir, result_dir, tfboard_dir = create_dir(args)
    logger = get_logger(log_dir, args.model, split='test')
    test(args)

    ShapeNetCars_dataset = build_ShapeNetCars()  # 5677 + 100 + 150
    CarsDataloader = torch.utils.data.DataLoader(ShapeNetCars_dataset, batch_size=1, shuffle=False, num_workers=8)
    Samples = [item for item in os.listdir(result_dir) if os.path.isdir(result_dir + '/' + item)]
    criterion = ChamferDistanceL2_split(ignore_zeros=True)
    get_Fidelity()
    get_Consistency()

    # get_MMD()
# CUDA_VISIBLE_DEVICES=0 python KITTI_metric.py     --vis=./experiments/PoinTr/KITTI_models/test_example/vis_result