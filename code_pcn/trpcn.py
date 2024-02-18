import argparse
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torch.optim as Optim
from torch.utils.data.dataloader import DataLoader

from dataset.shapenet import ShapeNetPcd
from models.utils_models import fps_subsample
from models.network import Model
from models.network_seedformer import idea_with_seedformer
from utils.metric import l1_cd, l2_cd
from utils.loss import cd_loss_L1
from utils.schedular import GradualWarmupScheduler
from utils.visualization import plot_pcd_one_view
from utils.utils import get_logger, create_dir, setup_seed

def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler

def train(params):
    # params = get_args()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    # 设置随机数种子
    setup_seed(1)

    log_dir, ckpt_dir, epochs_dir, _, tfboard_dir = create_dir(params)

    logger = get_logger(log_dir, params.model,split='train')
    logger.info(20*"="+"start"+20*"=")
    for key, val in params.__dict__.items():
        logger.info(f'params.{key} : {val}')

    logger.info('Loading Data...')
    train_dataset = ShapeNetPcd(dataroot='data/PCN', split='train', category = params.category)
    val_dataset = ShapeNetPcd(dataroot='data/PCN', split='val', category = params.category)
    train_dataloader = DataLoader(train_dataset,batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers )
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    logger.info("Dataset loaded!")

    # model
    device_ids = [0]
    model = idea_with_seedformer(up_factors=params.up_factors, fps_num=params.fps_num).cuda()
    logger.info(model)

    # optimizer
    optimizer = Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr, weight_decay=0, betas=(0.9, 0.999))        
    # lr scheduler
    scheduler_steplr = Optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=200,
                                          after_scheduler=scheduler_steplr)
     
    start_epoch = 0
    best_metrics = 1e8
    best_epoch = -1
    train_step, val_step = 0, 0
    # print(type(lr_scheduler))
    if params.resume:
        path_checkpoint = f"./{ckpt_dir}/ckpt-epoch-150.pth"
        checkpoint = torch.load(path_checkpoint)
        start_epoch = checkpoint['epoch']
        logger.info(f"Resuming from the {start_epoch +1} epoch !")

        best_metrics = checkpoint['best_metrics']
        logger.info(f"best_metrics: {best_metrics}")
        best_epoch = checkpoint['epoch']
        train_step = checkpoint['train_step'] 
        val_step = checkpoint['val_step']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.last_epoch = train_step

    nparams = sum([p.numel() for p in model.parameters()])
    logger.info(f"Params of model: {nparams}")    
    # model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    # training
    for epoch in range(start_epoch, params.epochs):

        model.train()
        epoch_start_time = time.time()
        batch_start_time = time.time()
       
        for idx, (p, c) in enumerate(train_dataloader):
            data_time = time.time() - batch_start_time
            partial, gt = p.cuda(), c.cuda()

            arr_pcd = model(partial)
            coarse, P1, P2, fine = arr_pcd
            # print(partial.shape, coarse.shape,P1.shape, P2.shape, fine.shape, gt.shape)
            gt_2 = fps_subsample(gt, P2.shape[1])
            gt_1 = fps_subsample(gt, P1.shape[1])
            gt_c = fps_subsample(gt_1, coarse.shape[1])
            
            cdc = cd_loss_L1(coarse, gt_c)
            cd1 = cd_loss_L1(P1, gt_1)
            cd2 = cd_loss_L1(P2, gt_2)
            cd3 = cd_loss_L1(fine, gt)

            loss_all = cdc + cd1 + cd2 + cd3  
            losses = [cdc, cd1, cd2, cd3]

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            batch_time = time.time() - batch_start_time
            batch_start_time = time.time()
            
            if (idx + 1) % 100 == 0:
                ss = f"Training Epoch [{epoch+1:03d}/{params.epochs:03d}] - Iteration [{ idx + 1:03d}/{len(train_dataloader):03d}]:  total loss = {loss_all * 1e3:.6f} [ "
                for j in range(0,len(losses)):
                    ss += f"{losses[j].item() * 1e3:.6f} " 
                
                t = f"] data_time = {data_time:.6f} batch_time = {batch_time:.6f}"
                logger.info(ss+t)
      
            if train_step <= 200:
                lr_scheduler.step()
               
            train_step += 1
        lr_scheduler.step()
        logger.info(f"epoch: {epoch+1}, lr: {optimizer.param_groups[0]['lr']}")

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        total_cd_l2 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization

            for i, (p, c) in enumerate(val_dataloader):
                partial, gt = p.to(params.device), c.to(params.device)

                arr_pcd = model(partial)
                coarse, p1, P2, fine = arr_pcd
                total_cd_l1 += l1_cd(fine, gt).item()
                total_cd_l2 += l2_cd(fine, gt).item()
                # save into image
                if rand_iter == i:
                    index = random.randint(0, fine.shape[0] - 1)
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_pcn{:03d}.png'.format(epoch)),
                                      [partial[index].detach().cpu().numpy(), arr_pcd[0][index].detach().cpu().numpy(), fine[index].detach().cpu().numpy(), gt[index].detach().cpu().numpy()],
                                      ['Input','Coarse','Output', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            
            total_cd_l1 /= len(val_dataset)
            total_cd_l2 /= len(val_dataset)
            val_step += 1

        epoch_end_time = time.time()
 
        logger.info(f"Validate Epoch [{epoch+1:03d}/{params.epochs:03d}]: L1 Chamfer Distance = {total_cd_l1 * 1e3:.6f} L2 Chamfer Distance = {total_cd_l2 * 1e4:.6f} Epoch time = {epoch_end_time - epoch_start_time:.3f}")
            
        if (epoch+1) % params.save_frequency == 0 or total_cd_l1 < best_metrics:
             
            if total_cd_l1 < best_metrics:
                file_name = 'ckpt-best.pth'
                best_metrics = total_cd_l1
                best_epoch = epoch+1
            else:
                file_name = f'ckpt-epoch-{epoch+1:03d}.pth'

            checkpoint_dict = {
                'epoch': epoch+1, 
                'train_step' : train_step,
                'val_step' : val_step,
                'best_metrics': best_metrics,
                'model_state_dict': model.state_dict(),
                'optim_state_dict' : optimizer.state_dict(),    
                'scheduler_state_dict':lr_scheduler
            }
            if len(device_ids) > 1:
                checkpoint_dict['model_state_dict'] = model.module.state_dict()    
            torch.save(checkpoint_dict, os.path.join(ckpt_dir, file_name))
            logger.info(f"Save checkpoint at {os.path.join(ckpt_dir, file_name)}")
            
    logger.info(f'Best l1 cd model in epoch {best_epoch}, the minimum l1 cd is {best_metrics * 1e3}')
    
    # train_writer.close()
    # val_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Training')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Tag of experiment')
    parser.add_argument('--datasets', type=str, default='ShapeNet', help='Datasets')
    parser.add_argument('--model', type=str, default='idea_with_fps_256', help='Logger directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--npoints', type=int, default=16384, help='number of gt point clouds')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=400, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--metric', type=str, default='cd_l1', help='mertic for Testing')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--resume', type=str, default=False, help='for resuming')
    parser.add_argument('--weights', type=str, default=False, help='pre-train')

    # model
    parser.add_argument('--up_factors', default=[1, 4, 8], help='for upsampling')
    parser.add_argument('--fps_num', type=int, default=256, help='fps_num')
    parser.add_argument('--save_frequency', type=int, default=50, help='Model saving frequency')
    
    params = parser.parse_args()
    train(params)