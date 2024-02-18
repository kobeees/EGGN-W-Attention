# conda activate /usr/local/miniconda3/envs/pth112env
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import open3d as o3d

import torch
import torch.utils.data as Data

from tqdm import tqdm
from dataset import shapenet
from models.network import Model
from models.network_seedformer import idea_with_seedformer
# from models.PCN import PCN

from utils.visualization import plot_pcd_one_view
from utils.metric import l1_cd, l2_cd, f_score
from utils.utils import make_dir,get_logger, create_dir

CATEGORIES_PCN       = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft']
CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']
# CATEGORIES_Completion3d  = ['plane', 'cabinet', 'car', 'chair', 'lamp', 'couch', 'table', 'watercraft']

def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)

def test_single_category(category, model, params, save=True):
    if save:
        cat_dir = os.path.join(params.result_dir, category)
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    test_dataset = shapenet.ShapeNetPcd(f'data/PCN', 'test_novel' if params.novel else 'test', category)
    # test_dataset = shapenet.ShapeNetPly(f'data/PCNDataset', 'test_novel' if params.novel else 'test', category)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    print(len(test_dataset), len(test_dataloader))

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    with torch.no_grad():
        for partial, gt in tqdm(test_dataloader):
    # for partial, gt in test_dataloader:
            partial, gt = partial.cuda(), gt.cuda()

            # for mine
            arr_pcd = model(partial)
            # coarse,fine = arr_pcd
            coarse  = arr_pcd[0]
            fine = arr_pcd[-1]

            total_l1_cd += l1_cd(fine, gt).item()
            total_l2_cd += l2_cd(fine, gt).item()
            # print(len(c))
            for i in range(len(gt)):
                coarse_pc = coarse[i].detach().cpu().numpy()
                # P1_pc = P1[i].detach().cpu().numpy()        
                # P2_pc = P2[i].detach().cpu().numpy()
                input_pc = partial[i].detach().cpu().numpy()
                output_pc = fine[i].detach().cpu().numpy()
                gt_pc = gt[i].detach().cpu().numpy()
                total_f_score += f_score(output_pc, gt_pc)
                if save:
                    plot_pcd_one_view(os.path.join(image_dir, '{:03d}_new.png'.format(index)), [input_pc,coarse_pc,output_pc, gt_pc], ['Input', 'coarse','Output', 'GT'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    # plot_pcd_one_view(os.path.join(image_dir, '{:03d}_new.png'.format(index)), [input_pc,coarse_pc,P1_pc, P2_pc,output_pc, gt_pc], ['Input', 'coarse','P1_pc','P2_pc','Output', 'GT'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
                    export_ply(os.path.join(output_dir, 'output_pc_{:03d}.ply'.format(index)), output_pc)
                    export_ply(os.path.join(output_dir, 'input_pc_{:03d}.ply'.format(index)), input_pc)
                    export_ply(os.path.join(output_dir, 'gt_pc_{:03d}.ply'.format(index)), gt_pc)
                index += 1

    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    avg_f_score = total_f_score / len(test_dataset)

    return avg_l1_cd, avg_l2_cd, avg_f_score

def test(params, save=False):
    log_dir, ckpt_dir, epochs_dir, result_dir, tfboard_dir = create_dir(params)
    if save:
        params.result_dir = result_dir
    logger = get_logger(log_dir, params.model,split='test')

    # load pretrained model mine
    model = idea_with_seedformer(up_factors=params.up_factors, fps_num=params.fps_num).cuda()
    path_checkpoint = f"./{ckpt_dir}/ckpt-best.pth"
    checkpoint = torch.load(path_checkpoint)
    
    # parameter resume of base model
    if checkpoint.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    elif checkpoint.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['base_model'].items()}
    elif checkpoint.get('model_state_dict') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    elif checkpoint.get('grnet') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['grnet'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    model.load_state_dict(base_ckpt)

    logger.info(f"Loading weights from {path_checkpoint}")
    logger.info(f"Best l1 cd model in epoch {checkpoint['epoch']},performance = 'CDL1': {checkpoint['best_metrics']*1e3}")
    # model = nn.DataParallel(model)

    model.eval()

    logger.info(20*"="+"Testing"+20*"=")
    logger.info('  {:20s}{:20s}{:20s}{:20s}'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)', 'FScore-0.01(%)'))
    logger.info('  {:20s}{:20s}{:20s}{:20s}'.format('--------', '-----------', '-----------', '--------------'))

    if params.category == 'all':
        if params.datasets=='ShapeNet':
            if params.novel:
                categories = CATEGORIES_PCN_NOVEL
            else:
                categories = CATEGORIES_PCN
        elif params.datasets == 'Completion3d':
            pass
            # categories = CATEGORIES_Completion3d
        else:
            print("Please choose right datasets!!")
        
        l1_cds, l2_cds, fscores = list(), list(), list()
        for category in categories:
            avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(category, model, params, save)
            logger.info('  {:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))
            l1_cds.append(avg_l1_cd)
            l2_cds.append(avg_l2_cd)
            fscores.append(avg_f_score)
        
        logger.info('  {:20s}{:20s}{:20s}{:20s}'.format('--------', '-----------', '-----------', '--------------'))
        logger.info('  {:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format('Average', np.mean(l1_cds) * 1e3, np.mean(l2_cds) * 1e4, np.mean(fscores) * 1e2))
    else:
        avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(params.category, model, params, save)
        logger.info('  {:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(params.category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Tag of experiment')
    parser.add_argument('--datasets', type=str, default='ShapeNet', help='Datasets')
    parser.add_argument('--npoints', type=int, default=16384, help='number of gt point clouds')
    parser.add_argument('--model', type=str, default='idea_with_fps_256', help='Logger directory')
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
    parser.add_argument('--fps_num', type=int, default=128, help='fps_num')

    params = parser.parse_args()

    test(params, params.save)

