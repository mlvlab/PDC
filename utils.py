import os
import torch
import numpy as np
import torch.distributed as dist
import random


device = torch.device('cuda')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('already exist')
        
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


##### Concerning DDP ##############
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0


def extract_weights(ofeature,sweight_cal,B,buffer = 1e-5):
    weight = ofeature+buffer
    if sweight_cal == 'normal':
        weight = torch.div(weight,torch.sum(weight,dim=2).view(B,-1,1))
    elif sweight_cal == 'softmax':
        weight = torch.nn.functional.softmax(weight,dim=2)
    
    if B == 1:
        weight = weight.squeeze(0)
    
    if True in torch.isnan(weight):
        import pdb; pdb.set_trace()
        
    return weight

