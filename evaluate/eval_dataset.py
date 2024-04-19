import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import random
import os
import json

class PartDataset(Dataset):
    def __init__(self, category, split_path, data_root, normalize_pc_max=1.03):
        self.category = category
        self.data_root = data_root
        self.normalize_pc_max = normalize_pc_max
        self.class2idx = {'plane':'02691156',
                          'chair':'03001627',
                          'table':'04379243',
                          'car':'02958343'}
        with open(split_path, 'r') as f:
            self.data_list = f.read().split('\n')

        self.seg_classes = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                            'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46],
                            'mug': [36, 37], 'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27],
                            'table': [47, 48, 49], 'plane': [0, 1, 2, 3], 'pistol': [38, 39, 40],
                            'chair': [12, 13, 14, 15], 'knife': [22, 23]}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        seed_everything(seed =42) ## for reproduction
        
        data_path = os.path.join(self.data_root, self.class2idx[self.category], self.data_list[index]+'.txt') 
        
        name = self.data_list[index]
        data = np.loadtxt(data_path).astype(np.float32)
        data[:,[2,0]] = data[:,[0,2]] 
        data[:,2] *= -1

        point_set = data[:, 0:3]
        seg = data[:, -1].astype(np.int32)
          
        point_set[:, 0:3] = self.normalize_pc(point_set[:, 0:3])


        return {'coords': torch.from_numpy(point_set), 
                'seg_label': torch.from_numpy(seg),
                'name': name}
    
    def normalize_pc(self, pc):
        bbox_x = (pc[:,0].max() + pc[:,0].min())/2
        bbox_y = (pc[:,1].max() + pc[:,1].min())/2
        bbox_z = (pc[:,2].max() + pc[:,2].min())/2
        bbox_centroid = np.array([bbox_x, bbox_y, bbox_z]).astype(np.float32)

        pc = pc - bbox_centroid
        dist = np.linalg.norm(pc, axis=1)
        pc /= (np.max(dist) * self.normalize_pc_max)

        return pc


class KeyPointDataset(Dataset):
    def __init__(self, category, split_path, data_root, normalize_pc_max=1.03):
        super().__init__()
        self.category = category
        self.normalize_pc_max = normalize_pc_max
        self.class2idx = {'plane':'02691156',
                          'chair':'03001627',
                          'table':'04379243',
                          'car':'02958343'}

        annots = json.load(open(os.path.join(data_root, 'annotations', category+".json")))
        annots = [annot for annot in annots if annot['class_id'] == self.class2idx[self.category]]
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])  
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        
        with open(split_path, 'r') as f:
            self.data_list = f.read().split('\n')

        self.pcds = []
        self.keypoints = []
        self.model_names = []
        self.idx2semids = []
        
        for fn in glob(os.path.join(data_root, 'pcds', self.class2idx[self.category], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in self.data_list:
                continue
            idx2semid = dict()
            curr_keypoints = -np.ones((self.nclasses,), dtype=np.int) 
            for i, kp in enumerate(keypoints[model_id]):
                curr_keypoints[kp[1]] = kp[0]
                idx2semid[i] = kp[1]
            self.keypoints.append(curr_keypoints)
            self.idx2semids.append(idx2semid)
            self.pcds.append(naive_read_pcd(fn)[0])
            self.model_names.append(model_id)

    def __getitem__(self, idx):
        seed_everything(seed =42) ## for reproduction

        pc = self.pcds[idx]
        label = self.keypoints[idx]
        model_name = self.model_names[idx]
        
        if self.normalize_pc_max:
            pc = self.normalize_pc(pc)
        
        return pc.astype(np.float32), label, model_name
        '''
            pc
                -> [2048,3] points (numpy list)
            label
                -> [semantic_label,] keypoint indices for each semantic label (numpy list)
            model_name
                -> object id (string)
        '''
    def __len__(self):
        return len(self.pcds)
    
    def normalize_pc(self, pc):
        bbox_x = (pc[:,0].max() + pc[:,0].min())/2
        bbox_y = (pc[:,1].max() + pc[:,1].min())/2
        bbox_z = (pc[:,2].max() + pc[:,2].min())/2
        bbox_centroid = np.array([bbox_x, bbox_y, bbox_z]).astype(np.float32)

        pc = pc - bbox_centroid
        dist = np.linalg.norm(pc, axis=1)
        pc /= (np.max(dist) * self.normalize_pc_max)

        return pc

def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
