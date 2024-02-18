import json
import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d

class ShapeNetPly(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """
    
    def __init__(self, dataroot, split, category='all'):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
            
            # alis for some seen categories
            "boat"      : "04530566",  # vessel
            "couch"     : "04256520",  # sofa
            "dresser"   : "02933112",  # cabinet
            "airplane"  : "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",

            # Completion3d
            # # seen categories
            # 'plane': '02691156',
            # 'cabinet': '02933112', 
            # 'car': '02958343', 
            # 'chair': '03001627', 
            # 'lamp': '03636649', 
            # 'couch': '04256520', 
            # 'table': '04379243', 
            # 'watercraft': '04530566',
            
            # # unseen categories
            # 'bench': '02828884',
            # 'monitor': '03211117', 
            # 'speaker': '03691459', 
            # 'firearm': '04090263', 
            # 'cellphone': '04401088', 
        }
        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}
        self.dataroot = dataroot
        self.split = split
        self.category = category

        self.partial_paths, self.gt_paths = self._load_data()
    
    def __getitem__(self, index):
        if self.split == 'train':
            partial_path = self.partial_paths[index].format(random.randint(0, 7))
        else:
            partial_path = self.partial_paths[index]
        complete_path = self.gt_paths[index]

        partial_pc = self.random_sample(self.read_point_cloud(partial_path), 2048)
        complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384)

        # print("partial_pc: ",partial_pc.shape ,"complete_pc: ",complete_pc.shape )

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.gt_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, f'{self.split}.list'), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        partial_paths, gt_paths = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            if self.split == 'train':
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '_{}.ply'))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.ply'))
            gt_paths.append(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.ply'))
        
        print(f'[DATASET] {len(gt_paths)} instances were loaded')
        return partial_paths, gt_paths
    
    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
    
    def random_sample(self, pc, n):
        # np.random.permutation() 总体来说他是一个随机排列函数,就是将输入的数据进行随机排列,
        # 官方文档指出,此函数只能针对一维数据随机排列,对于多维数据只能对第一维度的数据进行随机排列。
        # 简而言之：np.random.permutation函数的作用就是按照给定列表生成一个打乱后的随机列表
        # print("pc: ", pc.shape)
        idx = np.random.permutation(pc.shape[0])

        # print("idx: ", idx.shape)
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
            # print("idx_0: ", idx.shape)
        return pc[idx[:n]]

class ShapeNetPcd(data.Dataset):
    def __init__(self, dataroot, split, category='all'):
        assert split in ['train', 'val', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",

        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}
        # %02d的意思是如果输出的整型数不足两位，左侧用0补齐
        self.dataroot = dataroot
        self.npoints = 16384
        self.split = split
        self.dataset_categories = self.read_json('data/PCN/PCN.json')
        self.n_renderings = 8 if self.split == 'train' else 1
        self.category = category

        self.partial_paths, self.gt_paths = self._load_data()

        self.scale = 0
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __getitem__(self, index):
        if self.split == 'train':
            partial_path = self.partial_paths[index].format(random.randint(0, 7))
        else:
            partial_path = self.partial_paths[index]
        complete_path = self.gt_paths[index]

        partial_pc = self.random_sample(self.read_point_cloud(partial_path), 2048)
        complete_pc = self.random_sample(self.read_point_cloud(complete_path), 16384)

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.gt_paths)

    def _load_data(self):
        partial_paths, gt_paths = list(), list()
        if self.category=='all':
            for dc in self.dataset_categories:    
                # samples = random.sample(dc[self.split], len(dc[self.split])//3)        
                samples = dc[self.split]
                for model_id in samples:
                    if self.split == 'train':
                        partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', dc['taxonomy_id'], model_id, '{:02d}.pcd'))
                    else:
                        partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', dc['taxonomy_id'], model_id, '00.pcd'))
                    gt_paths.append(os.path.join(self.dataroot, self.split, 'complete', dc['taxonomy_id'], model_id + '.pcd'))
        else:
            for dc in self.dataset_categories:
                if self.category == dc["taxonomy_name"]:
                    samples = dc[self.split]
                    break
                else:
                    continue
            for model_id in samples:
                if self.split == 'train':
                    partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', dc['taxonomy_id'], model_id, '{:02d}.pcd'))
                else:
                    partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', dc['taxonomy_id'], model_id, '00.pcd'))
                gt_paths.append(os.path.join(self.dataroot, self.split, 'complete', dc['taxonomy_id'], model_id + '.pcd'))
        # if self.category != 'all':
        #     lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))

        print(f'[DATASET] {len(gt_paths)} instances were loaded')
        return partial_paths, gt_paths
    
    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
    
    def read_json(self, file_path):
        with open(file_path,'r',encoding='utf8') as fp:
            classmap = json.load(fp)
        return classmap

    def random_sample(self, pc, n):
        # np.random.permutation() 总体来说他是一个随机排列函数,就是将输入的数据进行随机排列,
        # 官方文档指出,此函数只能针对一维数据随机排列,对于多维数据只能对第一维度的数据进行随机排列。
        # 简而言之：np.random.permutation函数的作用就是按照给定列表生成一个打乱后的随机列表
        # print("pc: ", pc.shape)
        idx = np.random.permutation(pc.shape[0])

        # print("idx: ", idx.shape)
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
            # print("idx_0: ", idx.shape)
        return pc[idx[:n]]


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader
    train_dataset = ShapeNetPcd('data/PCN', 'train', category='car')
    val_dataset = ShapeNetPcd('data/PCN', 'val', category='car')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    print(f"length of train_dataloader: {len(train_dataloader)}")
    print(f"length of val_dataloader: {len(val_dataloader)}")
    # for i, data in enumerate(train_dataloader):
    #     partial = data[0]
    #     gt = data[1]
    #     # index = random.randint(0, gt.shape[0] - 1)
    #     plot_pcd_one_view(os.path.join('result', 'epoch_{:03d}.png'.format(i)),
    #                         [partial[0].detach().cpu().numpy(), gt[0].detach().cpu().numpy()],
    #                         ['Input','Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))