import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import einops

import modules
from meta_modules import HyperNetwork
from loss import *
import utils


class SAIT(nn.Module):
    def __init__(self, num_instances, latent_dim=256, num_branch = 12, model_type='sine', 
                 hyper_hidden_layers_D=3,hyper_hidden_features_D=256, 
                 hidden_features_D=256, num_hidden_layers_D=3,
                 hidden_features_T=128, num_hidden_layers_T =3, 
                 part_latent_dim=128, **kwargs):
        super().__init__()
        self.num_part = num_branch
        self.latent_dim = latent_dim
        self.part_latent_dim = part_latent_dim
        deform_in_features = 3 + self.part_latent_dim
        
        self.shape_latent_code = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.shape_latent_code.weight, mean=0, std=0.01)
        self.part_latent_basis = nn.Embedding(self.num_part, self.part_latent_dim)
        nn.init.normal_(self.part_latent_basis.weight, mean=0, std=0.01) 
        
        ## Deformation
        self.deform_net = modules.SingleBVPNet(type=model_type,mode='mlp', 
                        hidden_features=hidden_features_D, num_hidden_layers=num_hidden_layers_D, 
                        in_features=deform_in_features,out_features=4)
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_features=hyper_hidden_features_D,
                        hyper_hidden_layers=hyper_hidden_layers_D, hypo_module=self.deform_net)
        
        ## Template learning
        self.template_field = modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_features_T, 
                                num_hidden_layers=num_hidden_layers_T, in_features=3, out_features=1)
      
                
        # print(self)

    def soft_assignment(self,linear_w,B,N):
        pidxes = torch.tensor(range(self.num_part)).cuda()
        pidxes = pidxes.expand(B,N,self.num_part)
        pbases = self.part_latent_basis(pidxes)
            
        pbases = einops.rearrange(pbases,'b n w c -> (b n) c w')
        linear_w = einops.rearrange(linear_w,'b n w -> (b n) w 1')
        part_deformation_code = torch.einsum('b c w, b w k-> b c k',pbases,linear_w)
        part_deformation_code = einops.rearrange(part_deformation_code, '(b n) c 1-> b n c', b=B,n=N)
    
        return part_deformation_code


    def get_latent_code(self,instance_idx):
        embedding = self.shape_latent_code(instance_idx)
        return embedding
    
    # for generation
    def inference(self,coords,latent_z,embedding,extractor,sweight_cal='softmax'):
        with torch.no_grad():
            latent_o,_ = extractor(coords,latent_z)
            B,N,_ = latent_o.shape
            model_in = {'coords': coords}
            
            ## Deformation
            plabels = torch.max(latent_o, dim=2)[1].squeeze(0).detach().cpu().numpy()
            linear_w = utils.extract_weights(latent_o,sweight_cal,B)
            if B == 1:
                linear_w = linear_w.unsqueeze(0)
            part_deformation_code = self.soft_assignment(linear_w,B,N)
            model_in['part_deformation_code'] = part_deformation_code
            
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            deformation = model_output['model_out'][:,:,:3]
            correction = model_output['model_out'][:,:,3].unsqueeze(-1) 
            coords_dfm = coords + deformation
            
            #### Template learning ############################
            model_input_T = {'coords':coords_dfm}
            model_output_T = self.template_field(model_input_T)

            return model_output_T['model_out']+correction

    def get_template_coords(self,coords,latent_z,embedding,extractor,sweight_cal='softmax'):
        with torch.no_grad():
            latent_o,_ = extractor(coords,latent_z)
            B,N,_ = latent_o.shape
            model_in = {'coords': coords}
            
            ## Semantic Deformation
            plabels = torch.max(latent_o, dim=2)[1].squeeze(0).detach().cpu().numpy()
            linear_w = utils.extract_weights(latent_o,sweight_cal,B)
            if B == 1:
                linear_w = linear_w.unsqueeze(0)
            part_deformation_code = self.soft_assignment(linear_w,B,N)
            model_in['part_deformation_code'] = part_deformation_code
            
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)

            coords_dfm = coords + model_output['model_out'][:,:,:3]
            
            return coords_dfm

    def get_template_field(self,coords):
        model_in = {'coords': coords}
        with torch.no_grad():
            model_output = self.template_field(model_in)
            return model_output['model_out']

    # for training
    def forward(self, model_input, gt, Encoder, extractor, **kwargs):
        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] #.detach() # original coordinates
        B,N,_ = coords.shape
        
        ## Deformation
        # semantics extraction
        latent_z = Encoder(coords[:,:kwargs['on_surface_points']])
        latent_o,_ = extractor(coords,latent_z)
        plabels = torch.max(latent_o, dim=2)[1].squeeze(0).detach().cpu().numpy()
        model_out_final = {'plabels': plabels,'num_part': self.num_part}
        
        linear_w = utils.extract_weights(latent_o,kwargs['sweight_cal'],B)
        part_deformation_code = self.soft_assignment(linear_w,B,N)
        model_input['part_deformation_code'] = part_deformation_code

        embedding = self.shape_latent_code(instance_idx)
        hypo_params = self.hyper_net(embedding)
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output['model_out'][:,:,:3]
        correction = model_output['model_out'][:,:,3].unsqueeze(-1) 
        coords_dfm = coords + deformation

        # calculate gradient of the deformation field (for deformation smoothness)
        x = model_output['model_in'] # input coordinates -  p 
        grad_outputs = torch.ones_like(deformation[:,:,0])
        grad_u = torch.autograd.grad(deformation[:,:,0], [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(deformation[:,:,1], [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(deformation[:,:,2], [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_deform = torch.stack([grad_u,grad_v,grad_w],dim=2)  # gradient of deformation wrt. input position
        
        #### REGARDING parf dfm REGs ######################
        # extracting semantics of semantic deformed shapes
        latent_z_dfm = Encoder(coords_dfm[:,:kwargs['on_surface_points']])
        latent_o_dfm, _ = extractor(coords_dfm,latent_z_dfm)
        ###############################################################################################
        
        #### Template learning ############################
        model_input_T = {'coords':coords_dfm}
        model_output_T = self.template_field(model_input_T)
        sdf = model_output_T['model_out'] # SDF value in template space
        grad_temp = torch.autograd.grad(sdf, [coords_dfm], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in template space
        sdf_final = sdf + correction # add correction
        grad_sdf = torch.autograd.grad(sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in original shape space

        model_out = {'coords_ori':coords,'coords_dfm':coords_dfm, 
        'grad_deform': grad_deform,'sdf_correct':correction,
        'pred_sdf':sdf_final, 'grad_sdf':grad_sdf, 'grad_temp':grad_temp,
        'embedding': embedding, 'pdcode': part_deformation_code, 'latent_o_dfm':latent_o_dfm}

        model_out_final['info'] = model_out

        losses = SAIT_loss(model_out_final, gt, loss_grad_deform=kwargs['loss_grad_deform'],loss_grad_temp=kwargs['loss_grad_temp'],
        pair_permute=kwargs['pair_permute'],loss_correct=kwargs['loss_correct'],
        pdc_sem=kwargs['pdc_sem'],pdc_geo=kwargs['pdc_geo'],gdc_geo=kwargs['gdc_geo'],
        gl_scale=kwargs['gl_scale'],gl_scale_coef=kwargs['gl_scale_coef'])

        return losses
 
    # NEED TO MODIFY
    def embedding(self, embed, model_input, gt, Encoder, extractor, **kwargs):
        coords = model_input['coords'] # 3 dimensional input coordinates
        B,N,_ = coords.shape
        
        ## Deformation
        # semantics extraction
        latent_z = Encoder(coords[:,:kwargs['on_surface_points']])
        latent_o,_ = extractor(coords,latent_z)
        plabels = torch.max(latent_o, dim=2)[1].squeeze(0).detach().cpu().numpy()
        
        linear_w = utils.extract_weights(latent_o,kwargs['sweight_cal'],B)
        if B == 1:
            linear_w = linear_w.unsqueeze(0)

        part_deformation_code = self.soft_assignment(linear_w,B,N)
        model_input['part_deformation_code'] = part_deformation_code
        
        hypo_params = self.hyper_net(embed)
        model_output = self.deform_net(model_input, params=hypo_params)
        deformation = model_output['model_out'][:,:,:3]  # 3 dimensional deformation field
        correction = model_output['model_out'][:,:,3:] # scalar correction field

        coords_dfm = coords + deformation
        model_input_T = {'coords':coords_dfm}

        ## Geometric Template
        model_output_Tg= self.template_field(model_input_T)
        sdf = model_output_Tg['model_out'] # SDF value in template space
        sdf_final = sdf + correction # add correction
        x = model_output['model_in'] # input coordinates -  p 
        grad_sdf = torch.autograd.grad(sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in original shape space
        
        model_out = {'coords_ori':coords,'coords_dfm':coords_dfm,'pred_sdf':sdf_final, 'embedding':embed, 'grad_sdf':grad_sdf}
        
        model_out_final = {}
        model_out_final['info'] = model_out
        losses = SAIT_loss_TTO(model_out_final, gt)

        return losses



class encoder_branch(nn.Module):
    def __init__(self, z_dim, point_dim, num_branch,num_hidden_layer=3,L1_dim=3072,L2_dim=384):
        super(encoder_branch, self).__init__()
        
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.num_hidden_layer = num_hidden_layer
        self.L1_dim = L1_dim
        self.L2_dim = L2_dim
        
        if self.num_hidden_layer == 3:
            self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.L1_dim, bias=True)
            self.linear_2 = nn.Linear(self.L1_dim, self.L2_dim, bias=True)
            self.linear_3 = nn.Linear(self.L2_dim, num_branch) ## num_branch
            
            nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.linear_1.bias,0)
            nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.linear_2.bias,0)
            nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.linear_3.bias,0)
        
        elif self.num_hidden_layer == 2:
            self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.L1_dim, bias=True)
            self.linear_2 = nn.Linear(self.L1_dim, num_branch, bias=True)
            
            nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.linear_1.bias,0)
            nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.linear_2.bias,0)

    
    def forward(self, points, z):
        zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
        pointz = torch.cat([points,zs],2)
        
        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        if self.num_hidden_layer == 3:
            l2 = self.linear_2(l1)
            l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

            l3 = self.linear_3(l2)
            l3 = torch.sigmoid(l3)
            l4 = F.max_pool1d(l3, l3.shape[2])
        
            return l3, l4
        
        elif self.num_hidden_layer == 2:
            l2 = self.linear_2(l1)
            l2 = torch.sigmoid(l2)
            l3 = F.max_pool1d(l2, l2.shape[2])
        
            return l2, l3

class encoder_global(nn.Module):
    def __init__(self, channel=3, z_dim=256):
        super(encoder_global, self).__init__()

        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)       
        self.fc2 = nn.Linear(512,  z_dim)
        self.relu = nn.ReLU(inplace=True)
                
    def forward(self, x):
        # x:  B*num_points*3
        x = x.transpose(2,1).contiguous()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class encoder(nn.Module):
    def __init__(self, z_dim, num_branch=12, point_dim = 3,num_hidden_layer=3,L1_dim=3072,L2_dim=384):
        super(encoder, self).__init__()
        
        self.z_dim = z_dim
        self.num_branch = num_branch
        self.point_dim = point_dim
        self.num_hidden_layer = num_hidden_layer
        self.L1_dim = L1_dim
        self.L2_dim = L2_dim
        self.enc_global = encoder_global(z_dim=self.z_dim)
        self.enc_branch = encoder_branch(self.z_dim, self.point_dim, self.num_branch,self.num_hidden_layer, self.L1_dim, self.L2_dim)
        
    
    def forward(self, model_input):
        surface = model_input['surface']
        coords = model_input['query_coords']

        z_vector = self.enc_global(surface)
        branch, net_out = self.enc_branch(coords, z_vector)
        
        return z_vector, branch, net_out
    
## Original DIF-Net
class DeformedImplicitField(nn.Module):
    def __init__(self, num_instances,latent_dim=128, model_type='sine', 
            hyper_hidden_layers_D=3,hyper_hidden_features_D=256, 
            hidden_features_D=256, num_hidden_layers_D=3,
            hidden_features_T=128, num_hidden_layers_T =3,**kwargs):
        super().__init__()
        #　We use auto-decoder framework following Park et al. 2019 (DeepSDF),
        # therefore, the model consists of latent codes for each subjects and DIF-Net decoder.

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        ## DIF-Net
        # template field
        self.template_field = modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_features_T, 
                            num_hidden_layers=num_hidden_layers_T, in_features=3,out_features=1)

        # Deform-Net
        self.deform_net=modules.SingleBVPNet(type=model_type,mode='mlp', hidden_features=hidden_features_D, 
                                num_hidden_layers=num_hidden_layers_D, in_features=3,out_features=4)

        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=self.latent_dim, hyper_hidden_features=hyper_hidden_features_D,
                        hyper_hidden_layers=hyper_hidden_layers_D, hypo_module=self.deform_net)
        # print(self)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self,instance_idx):
        embedding = self.latent_codes(instance_idx)

        return embedding

    # for generation
    def inference(self,coords,embedding):

        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)

            deformation = model_output['model_out'][:,:,:3]
            correction = model_output['model_out'][:,:,3:]
            new_coords = coords + deformation
            model_input_temp = {'coords':new_coords}
            model_output_temp = self.template_field(model_input_temp)

            return model_output_temp['model_out'] + correction

    def get_template_coords(self,coords,embedding,deltas=False):
        with torch.no_grad():
            model_in = {'coords': coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            deformation = model_output['model_out'][:,:,:3]
            new_coords = coords + deformation
            if deltas:
                return new_coords, model_output['model_out'][:,:,-1]
            else:
                return new_coords

    def get_template_field(self,coords):
        with torch.no_grad():
            model_in = {'coords': coords}
            model_output = self.template_field(model_in)

            return model_output['model_out']

    # for training
    def forward(self, model_input,gt,**kwargs):

        instance_idx = model_input['instance_idx']
        coords = model_input['coords'] # 3 dimensional input coordinates

        # get network weights for Deform-net using Hyper-net 
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)

        # [deformation field, correction field]
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output['model_out'][:,:,:3]  # 3 dimensional deformation field
        correction = model_output['model_out'][:,:,3:] # scalar correction field
        new_coords = coords + deformation # deform into template space

        # calculate gradient of the deformation field
        x = model_output['model_in'] # input coordinates
        u = deformation[:,:,0]
        v = deformation[:,:,1]
        w = deformation[:,:,2]

        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(w, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_deform = torch.stack([grad_u,grad_v,grad_w],dim=2)  # gradient of deformation wrt. input position

        model_input_temp = {'coords':new_coords}

        model_output_temp = self.template_field(model_input_temp)

        sdf = model_output_temp['model_out'] # SDF value in template space
        grad_temp = torch.autograd.grad(sdf, [new_coords], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in template space

        sdf_final = sdf + correction # add correction

        grad_sdf = torch.autograd.grad(sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in original shape space
        model_out = {'model_in':model_output['model_in'],'grad_temp':grad_temp,'grad_deform':grad_deform, 'model_out':sdf_final, 'latent_vec':embedding,
                'hypo_params':hypo_params,'grad_sdf':grad_sdf,'sdf_correct':correction,
                'coords':coords[:,:kwargs['on_surface_points']],'coords_dfm':new_coords[:,:kwargs['on_surface_points']]}
        losses = deform_implicit_loss(model_out, gt,loss_grad_deform=kwargs['loss_grad_deform'],loss_grad_temp=kwargs['loss_grad_temp'],loss_correct=kwargs['loss_correct'])

        return losses

    # for evaluation
    def embedding(self, embed, model_input,gt):

        coords = model_input['coords'] # 3 dimensional input coordinates

        # get network weights for Deform-net using Hyper-net 
        hypo_params = self.hyper_net(embed)

        # [deformation field, correction field]
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output['model_out'][:,:,:3] # 3 dimensional deformation field
        correction = model_output['model_out'][:,:,3:] # scalar correction field
        new_coords = coords + deformation # deform into template space

        model_input_temp = {'coords':new_coords}

        model_output_temp = self.template_field(model_input_temp)

        sdf = model_output_temp['model_out'] # SDF value in template space
        sdf_final = sdf + correction # add correction

        x = model_output['model_in'] # input coordinates
        grad_sdf = torch.autograd.grad(sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True)[0] # normal direction in original shape space

        model_out = {'model_in':model_output['model_in'], 'model_out':sdf_final, 'latent_vec':embed, 'grad_sdf':grad_sdf}
        losses = embedding_loss(model_out, gt)

        return losses
    
    
# if wmix==0:
            #     latent_o = torch.empty(latent_o.shape).normal_(mean=0,std=2).cuda()
            # elif wmix==1:
            #     tmp = torch.zeros(latent_o[0].shape)
            #     tmpidx = torch.zeros([N])
            #     for i in range(N):
            #         tmpidx[i] = torch.randperm(6)[0]
            #     tmpidx = tmpidx.unsqueeze(-1)
            #     tmp = tmp.scatter(1, tmpidx.long(), 1)
            #     latent_o = tmp.unsqueeze(0).cuda()
            # elif wmix==2:
            #     tmp = torch.zeros(latent_o[0].shape)
            #     tmp[:,3] = 1
            #     latent_o = tmp.unsqueeze(0).cuda()
            # elif wmix==3:
            #     tmp = torch.zeros(latent_o[0].shape)
            #     tmp[:,4] = 1
            #     latent_o = tmp.unsqueeze(0).cuda()
            # elif wmix==5:
            #     tmp = torch.zeros(latent_o[0].shape)
            #     tmp[:,5] = 1
            #     latent_o = tmp.unsqueeze(0).cuda()