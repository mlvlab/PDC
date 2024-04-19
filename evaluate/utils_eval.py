import os
import torch
import numpy as np
import einops
import torch.nn.functional as F
import math
import itertools
import time


class_part_label = {'chair':4,'table':2,'car':4,'plane':4}


def get_center_keypoints(path_kp, src_id):
    num_kps =  []
    for instance in src_id:
        num_kps.append(len(torch.load(os.path.join(path_kp, instance, 'kp_label.pt'))))
    num_kp = max(num_kps)
    seg_nums = np.array([0]*num_kp, dtype='int32')  
    kps_sum = np.stack([[0.0,0.0,0.0]]*num_kp,axis=0)
    
    for instance in src_id:
        kp_coord = np.stack([[0.0,0.0,0.0]]*num_kp, axis=0)
        deformed_coords = torch.load(os.path.join(path_kp, instance, 'coords.pt')).numpy()[:,4:]
        kp_label = torch.load(os.path.join(path_kp, instance, 'kp_label.pt')).numpy()
        mask = (kp_label != -1)
        seg_nums += mask
  
        t = deformed_coords[kp_label]
        kp_coord[mask] = t[mask]
        kps_sum += kp_coord
        
    kp_center = kps_sum/ seg_nums[:,None]
    return kp_center

def get_deformed_points(path_kp, id_list):
    src_o = []
    src_d = []
    gt_kp_indices = []
    
    name_list = []
    for instance in id_list:
        name_list.append(instance)
        coords = torch.load(os.path.join(path_kp, instance, 'coords.pt')).numpy()[:,:3]

        deformed_coords = torch.load(os.path.join(path_kp, instance, 'coords.pt')).numpy()[:,4:]
        kp_label = torch.load(os.path.join(path_kp, instance, 'kp_label.pt')).numpy()

        src_o.append(coords) # original points
        src_d.append(deformed_coords) 
        gt_kp_indices.append(kp_label)

    return src_o, src_d, gt_kp_indices, name_list
    

def part_label_transfer(spath, src_id, tgt_ids, category, bsize=5, k=10, tau=3):
    tgt_num = len(tgt_ids)
    ious,plist,glist,ilist = [],[],[],[]
    tgt_o,tgt_d,tgt_gt,tgt_lp = [],[],[],[]
    info = {}
    pws0, nidxs0 = [],[]
    
    # Extracting Source shape info
    if type(src_id) == list:
        src_dfm = torch.zeros([0])
        src_label = torch.zeros([0])
        for sid in src_id:
            src_set = torch.load(os.path.join(spath,sid,'coords.pt'))
            src_dfm = torch.cat([src_dfm,src_set[:,4:7]],dim=0)
            src_label = torch.cat([src_label,src_set[:,3]])
    else: # one-shot transfer
        src_dfm = torch.load(os.path.join(spath,src_id,'coords.pt'))[:,4:7]
        src_label = torch.load(os.path.join(spath,src_id,'coords.pt'))[:,3]
    src_label_len = class_part_label[category]
    if src_label_len == len(torch.unique(src_label)):
        ## 1. Extract point-wise weights
        t1 = time.time()
        crit = list(range(0,tgt_num,bsize))
        crit.append(tgt_num)
        for i in range(math.ceil(tgt_num/bsize)):
            tgt_id = tgt_ids[crit[i]:crit[i+1]]
            tmp1,tmp2 = voting_pointwise(spath,src_id,tgt_id,k,tau)
            pws0.append(tmp1)
            nidxs0.append(tmp2)
        pws = list(itertools.chain(*pws0))
        nidx = list(itertools.chain(*nidxs0))
#         print(time.time()-t1)
        ## 2. Calculating IoU
        for t, tid in enumerate(tgt_ids):
            sav_dfm = torch.load(os.path.join(spath,tid,'coords.pt')) # 3-1-3-3: ori, part label, dfm(pretrained), dfm(ours)
            tgt_o.append(sav_dfm[:,:3])
            tgt_d.append(sav_dfm[:,4:7])
            tgt_gt.append(sav_dfm[:,3])
            
            gt = sav_dfm[:,3]
            seg_classes = list(np.unique(gt.numpy()).astype(np.int64))
            if category == 'table':
                seg_classes = [47,48]

            pw_weight = pws[t]
            nearest_idx = nidx[t]
            gt = gt.type(torch.LongTensor)
            # squeeze
            src_label_sq = src_label[nearest_idx].view(-1)
            src_label_ohv = F.one_hot((src_label_sq-torch.min(src_label)).long(),num_classes=src_label_len)
            src_label_ohv = einops.rearrange(src_label_ohv,'(p l) c -> p l c',l=k)

            pred_wsum = pw_weight * src_label_ohv.to('cuda')
            pred = torch.argmax(torch.sum(pred_wsum,dim=1),dim=-1)+torch.min(src_label)
            pred = pred.type(torch.LongTensor)
            
            tgt_lp.append(pred.detach().cpu())
            # get histograms
            inter = pred[(pred==gt).nonzero(as_tuple=True)] # True Positive
            inter_hg = torch.bincount(inter, minlength=seg_classes[-1]+1)[seg_classes]  # (num_class) TP histogram for each class
            pred_hg = torch.bincount(pred, minlength=seg_classes[-1]+1)[seg_classes] # (num_class) prediction histogram for each class 
            gt_hg = torch.bincount(gt, minlength=seg_classes[-1]+1)[seg_classes]  # (num_class)    gt histogram for each class 
            union_hg = pred_hg + gt_hg - inter_hg                                    # (num_class)
            
            iou_class = inter_hg / (union_hg + 1e-10)                                # IoU for each class
            acc_class = inter_hg / (gt_hg + 1e-10)                               # Accuracy for each class
            mIoU = torch.mean(iou_class)
            mAcc = torch.mean(acc_class)

            ious.append(mIoU.item())
            plist.append(pred_hg.numpy())
            glist.append(gt_hg.numpy())
            ilist.append(inter_hg.numpy())
        info['pred'] = plist
        info['gt'] = glist
        info['inter'] =ilist

        return src_dfm,src_label,tgt_o, tgt_d, tgt_gt, tgt_lp,ious,info
    else:
        return None


def voting_pointwise(spath,src_id,tgt_ids,k,tau):
    tgt_summed = torch.zeros([0])
    tgt_len = [0]
    
    for i, tid in enumerate(tgt_ids):
        tgt_set = torch.load(os.path.join(spath,tid,'coords.pt'))
        tgt_len.append(tgt_set.shape[0])
        tgt_summed = torch.cat([tgt_summed,tgt_set[:,4:7]],dim=0)
    if type(src_id) == list:
        src_dfm = torch.zeros([0])
        src_label = torch.zeros([0])
        for sid in src_id:
            src_set = torch.load(os.path.join(spath,sid,'coords.pt'))
            src_dfm = torch.cat([src_dfm,src_set[:,4:7]],dim=0)
            src_label = torch.cat([src_label,src_set[:,3]])
    else: # one-shot transfer
        src_dfm = torch.load(os.path.join(spath,src_id,'coords.pt'))[:,4:7]
        src_label = torch.load(os.path.join(spath,src_id,'coords.pt'))[:,3]
    src_label_len = len(torch.unique(src_label))

    dist_summed = torch.cdist(src_dfm.unsqueeze(0).cuda(),tgt_summed.unsqueeze(0).cuda(),p=2)
    tgt_pweights = []
    tgt_nidxes = []
    stt = 0
    
    # Extract pw_weight & nearest_idx
    for b in range(len(tgt_len)-1):
        stt += tgt_len[b]
        dist = dist_summed[...,stt:stt+tgt_len[b+1]]
        dist = einops.rearrange(dist,'b S T -> T (b S)')
        dist_slt, nearest_idx = torch.topk(dist, dim=1, k=k, largest=False)
        dist_slt = torch.exp(-tau*dist_slt)
        pw_weight = (dist_slt)/(torch.sum(dist_slt,dim=1,keepdim=True))
        pw_weight = pw_weight.unsqueeze(-1).repeat(1,1,src_label_len)
        tgt_pweights.append(pw_weight) # .detach().cpu()
        tgt_nidxes.append(nearest_idx)
        
    return tgt_pweights, tgt_nidxes
