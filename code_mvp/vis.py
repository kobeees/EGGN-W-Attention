import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import open3d as o3d
import importlib
import argparse
import munch
import yaml
from tqdm import tqdm
from models_vis.pcn import PCN
from models_vis.pmpnet import PMPNet, PMPNetPlus
from models_vis.snowflakenet import SnowflakeNet
from models_vis.topnet import TopNet
from models_vis.newfast import Newfast

from utils.train_utils import *
from dataset import ShapeNetH5

import matplotlib.pyplot as plt

def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30  # 水平倾斜
    azim = -45  # 旋转
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"The directory created successfully at {dir_path}")
    else:
        print(f"The {dir_path} already exists! ")

def test():
    dataset_test = ShapeNetH5(train=False, npoints=args.num_points)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    # load pretrained model  PCN
    model_pcn = PCN(args).cuda()
    path_checkpoint = f"./log/vis/pcn_best_cd_t.pth"
    # model_pcn.load_state_dict(torch.load(path_checkpoint))
    model_pcn.load_state_dict(torch.load(path_checkpoint)['net_state_dict'])

    # topnet
    model_topnet = TopNet(args).cuda()
    path_checkpoint = f"./log/vis/topnet_best_cd_t.pth"
    # model_pcn.load_state_dict(torch.load(path_checkpoint))
    model_topnet.load_state_dict(torch.load(path_checkpoint)['net_state_dict'])

    # pmpnet
    model_pmpnet = PMPNet(args).cuda()
    path_checkpoint = f"./log/vis/pmpnet_best_cd_t.pth"
    # model_pcn.load_state_dict(torch.load(path_checkpoint))
    model_pmpnet.load_state_dict(torch.load(path_checkpoint)['net_state_dict'])

    # pmpnet++
    model_pmpnetplus = PMPNetPlus(args).cuda()
    path_checkpoint = f"./log/vis/pmpnetplus_best_cd_t.pth"
    # model_pcn.load_state_dict(torch.load(path_checkpoint))
    model_pmpnetplus.load_state_dict(torch.load(path_checkpoint)['net_state_dict'])

    # snowflakenet
    model_snow = SnowflakeNet(args).cuda()
    path_checkpoint = f"./log/vis/snowflake_best_cd_t.pth"
    # model_pcn.load_state_dict(torch.load(path_checkpoint))
    model_snow.load_state_dict(torch.load(path_checkpoint)['net_state_dict'])

    # load pretrained model mine
    model_new = Newfast(args).cuda()
    path_checkpoint = f"./log/vis/newfast_best_cd_t.pth"
    # model_pcn.load_state_dict(torch.load(path_checkpoint))
    model_new.load_state_dict(torch.load(path_checkpoint)['net_state_dict'])


    models = [model_pcn,  model_topnet, model_pmpnet, model_pmpnetplus, model_snow, model_new]
    models_name = ['pcn', 'topnet', 'pmpnet', 'pmpnetplus', 'snowflakenet', 'newfast']
    #
    for model in models:
        model.eval()


    cat_names = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft',
                'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
    idx_to_plot = [i for i in range(0, 41600, 1)]

    logging.info('Testing...')
    if args.save_vis:
        for category in cat_names:
            cat_dir = os.path.join(log_dir, category)
            image_dir = os.path.join(cat_dir, 'image')
            make_dir(cat_dir)
            make_dir(image_dir)

            for name in models_name:
                output_dir = os.path.join(cat_dir, name)
                make_dir(output_dir)

    with torch.no_grad():
        # for data in tqdm(dataloader_test):
        for i, data in enumerate(tqdm(dataloader_test)):

            label, inputs_cpu, gt_cpu = data

            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            partial = inputs.transpose(2, 1).contiguous()

            fines = []
            for model in models:
                fine = model(partial, gt, is_training=False)

                fines.append(fine)

            # l = 0
            # for item in fines:
            #     print(l, item.shape)
            #     l = l + 1

            if args.save_vis:
                for j in range(args.batch_size):
                    idx = i * args.batch_size + j
                    # if idx in idx_to_plot:
                    pic = f'{cat_names[int(label[j])]}_{idx}.png'
                    plyname = f'{cat_names[int(label[j])]}_{idx}.ply'
                    # print('object_%d.png' % idx)
                    fine_pcn = fines[0][j].detach().cpu().numpy()
                    fine_topnet = fines[1][j].detach().cpu().numpy()
                    fine_pmpnet = fines[2][j].detach().cpu().numpy()
                    fine_pmpnetplus = fines[3][j].detach().cpu().numpy()
                    fine_snowflakenet = fines[4][j].detach().cpu().numpy()
                    fine_newfast = fines[5][j].detach().cpu().numpy()
                    input_pc = inputs[j].detach().cpu().numpy()
                    gt_pc = gt[j].detach().cpu().numpy()

                    save_path = os.path.join(log_dir, cat_names[int(label[j])])

                    plot_pcd_one_view(os.path.join(save_path, 'image', pic),
                                        [input_pc, fine_pcn, fine_topnet, fine_pmpnet, fine_pmpnetplus, fine_snowflakenet, fine_newfast,gt_pc],
                                    ['partial', 'pcn','topnet', 'pmpnet','pmpnetplus','snowflakenet','Ours', 'GT'],
                                    xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    for j, name in enumerate(models_name):
                        export_ply(os.path.join(save_path, name, f'{name}_{plyname}'), eval(f'fine_{name}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.dirname(args.load_model)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
