import os
import time
import numpy as np
import torch

from .Logger import Log

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True  # 为了确定算法，保证得到一样的结果。
    torch.backends.cudnn.enabled = True  # 使用非确定性算法
    # 设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
    # 当这个参数设置为True时，启动算法的前期会比较慢，但算法跑起来以后会非常快。
    torch.backends.cudnn.benchmark = True  # 是否自动加速。

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"The directory created successfully at {dir_path}")
    else:
        print(f"The {dir_path} already exists! ")

def get_logger(params, model_name, split):
    # timestamp = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    logs = Log(params, model_name, split)
    logger = logs.get_log()
    return logger  

def create_dir(params):
    # prepare logger directory
    log_dir = os.path.join(params.exp_name, params.model,params.datasets)
    log_dir += f"_{params.npoints}_lr_{params.lr}_metric_{params.metric}"
    make_dir(log_dir)
    tt = time.strftime('%Y-%m-%d_%H-%M-%S')
    tfboard_dir = os.path.join(log_dir, f"tfboard/{tt}")
    ckpt_dir = os.path.join(log_dir,'checkpoints')
    epochs_dir = os.path.join(log_dir, 'epochs')
    result_dir = os.path.join(log_dir, 'result')
    
    make_dir(tfboard_dir)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)
    make_dir(result_dir)

    return log_dir, ckpt_dir, epochs_dir, result_dir, tfboard_dir


def random_dropping(pc, e):
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1, 1))[0, 0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim=1)
    return pc
