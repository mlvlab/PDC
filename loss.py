import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss import chamfer_distance
import itertools
import utils


l2_loss = nn.MSELoss()


def SAIT_loss(model_output, gt, loss_grad_deform=1,loss_grad_temp=100,pdc_sem=10,pdc_geo=10,gdc_geo=10,
            loss_correct=1e2,gl_scale=0,gl_scale_coef=1,pair_permute='False',dist_lambda=5):
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    plabels = model_output['plabels']
    num_part = model_output['num_part']
    coords_ori = model_output['info']['coords_ori']
    B, N, _ = coords_ori.shape
    coords_dfm = model_output['info']['coords_dfm'] 
    latent_o_dfm = model_output['info']['latent_o_dfm']
    pred_sdf = model_output['info']['pred_sdf']
    gradient_sdf = model_output['info']['grad_sdf']
    sdf_correct = model_output['info']['sdf_correct']
    
    gradient_deform = model_output['info']['grad_deform']
    gradient_temp = model_output['info']['grad_temp']
    
    if utils.str2bool(pair_permute):
        pairs = list(itertools.combinations(range(B), 2)) # B' (B C 2)
    else:
        pairs = np.array_split(torch.randperm(B), 2) # B/2

    ### Reconstruction ###
    sdf_constraint = torch.where(gt_sdf != -1, torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient_sdf[..., :1]))
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)
    
    grad_temp_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_temp, gt_normals, dim=-1)[..., None],
                                torch.zeros_like(gradient_temp[..., :1]))

    # global scaling regularization
    scale_nmt = coords_dfm.unsqueeze(-2) @ coords_ori.unsqueeze(-1)
    scale_dmt = coords_ori.unsqueeze(-2) @ coords_ori.unsqueeze(-1)
    if gl_scale == 1:
        glo_s = torch.abs(scale_nmt.sum()/scale_dmt.sum()-1)
    elif gl_scale == 2:
        glo_s = torch.abs((scale_nmt.sum(dim=1)/scale_dmt.sum(dim=1)).mean()-1)
    else:
        glo_s = 0.
    ### Deformation smoothness ###
    # constraining deformation gradient
    grad_deform_constraint = gradient_deform.norm(dim=-1)
    # minimal correction prior
    sdf_correct_constraint = torch.abs(sdf_correct)
    # initialization
    wshape_dfm1 = torch.zeros(0).cuda()
    wshape_dfm2 = torch.zeros(0).cuda()
    pdc_sem_1,pdc_sem_2,pdc_geo_1,pdc_geo_2 = torch.tensor(0.).cuda(),torch.tensor(0.).cuda(),torch.tensor(0.).cuda(),torch.tensor(0.).cuda()
    
    ### Part + Global deformation consistency regularizations #####
    for pair in pairs:
        # gdc_geo: global deformation consistency on data space
        wshape_dfm1 = torch.cat([wshape_dfm1,coords_dfm[pair[0]].unsqueeze(0)]) # B'x N x 3
        wshape_dfm2 = torch.cat([wshape_dfm2,coords_dfm[pair[1]].unsqueeze(0)])
        
        ## PAIRWISE - for part deformation consistency
        for branch in list(range(num_part)):
            label_picked1 = np.where(plabels[pair[0]] == branch)[0]
            label_picked2 = np.where(plabels[pair[1]] == branch)[0]

            if len(label_picked1) != 0 and len(label_picked2) != 0:
                shape_dfm1 = coords_dfm[pair[0]][label_picked1].unsqueeze(0) # 1xN1x3
                shape_dfm2 = coords_dfm[pair[1]][label_picked2].unsqueeze(0) # 1xN2x3
                sem_dfm1 = latent_o_dfm[pair[0]][label_picked1].unsqueeze(0) # 1xN1xk
                sem_dfm2 = latent_o_dfm[pair[1]][label_picked2].unsqueeze(0) # 1xN2xk
                # Calculating part correspondence (euclidean space + feature space)
                xyz_dist = torch.sqrt(torch.sum((shape_dfm1.squeeze(0)[None,:,:] - shape_dfm2.squeeze(0)[:,None,:])**2, axis=-1))
                feature_dist = torch.sqrt(torch.sum((sem_dfm1.squeeze(0)[None,:,:] - sem_dfm2.squeeze(0)[:,None,:])**2, axis=-1))
                dist = xyz_dist + (dist_lambda*feature_dist)
                corr1 = torch.argmin(dist, axis = 0).unsqueeze(0)
                corr2 = torch.argmin(dist, axis = 1).unsqueeze(0)

                # part deformation consistency on data space (p+vs)
                if pdc_geo != 0:
                    pdc_geo_1 += l2_loss(shape_dfm1,shape_dfm2[0][corr1.view(-1)].unsqueeze(0).detach())
                    pdc_geo_2 += l2_loss(shape_dfm2,shape_dfm1[0][corr2.view(-1)].unsqueeze(0).detach())
                       
                # part deformation consistency on feature space (p+vs)
                if pdc_sem != 0:
                    pdc_sem_1 += l2_loss(sem_dfm1,sem_dfm2[0][corr1.view(-1)].unsqueeze(0).detach())
                    pdc_sem_2 += l2_loss(sem_dfm2,sem_dfm1[0][corr2.view(-1)].unsqueeze(0).detach())

    if gdc_geo != 0: 
        gdc_geo_loss = chamfer_distance(wshape_dfm1,wshape_dfm2)[0]
    else:
        gdc_geo_loss = 0.
        
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3, 
            'inter': inter_constraint.mean() * 5e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1,
            'grad_deform_constraint':grad_deform_constraint.mean()* loss_grad_deform,
            'grad_temp_constraint': grad_temp_constraint.mean() * loss_grad_temp,
            'sdf_correct_constraint':sdf_correct_constraint.mean()* loss_correct,
            'embedding_constraint': torch.mean(model_output['info']['embedding'] ** 2).mean() * 1e6,
            'pdcode_constraint': torch.mean(model_output['info']['pdcode'] ** 2).mean() * 1e6,
            'glo_scale': glo_s*gl_scale_coef,
            'pdc_sem': (pdc_sem_1/B+pdc_sem_2/B)*pdc_sem,
            'gdc_geo': gdc_geo_loss*gdc_geo,
            'pdc_geo': (pdc_geo_1/B+pdc_geo_2/B)*pdc_geo}
            
