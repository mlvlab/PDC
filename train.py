import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import configargparse
import wandb
import io

import torch
from torch.utils.data import DataLoader
import dataset, utils, training_loop
from model import SAIT, encoder

import pdb

p = configargparse.ArgumentParser()
p.add_argument('--use_wandb', type=bool, default=False)
p.add_argument('--project_title', type=str, default='XXX')
p.add_argument('--config', type=str, default='configs/train/XXX.yml')
p.add_argument('--gpu', type=str, default='7')
p.add_argument('--experiment_name', type=str, default='XXX') 

#### tuning parameters ###################################################
p.add_argument('--lr', type=float, default=0.0001)

# regarding deformation
p.add_argument('--loss_grad_deform', type=float,default=10, help='deformation smoothness prior.')
p.add_argument('--loss_grad_temp', type=float,default=100, help='loss weight for normal consistency prior.')
p.add_argument('--loss_correct', type=float,default=100)
p.add_argument('--pdc_sem', type=float, default=500)
p.add_argument('--pdc_geo', type=float, default=4000)
p.add_argument('--gdc_geo', type=float, default=50)
###########################################################################
p.add_argument('--assign_latp',type=str,default='soft',help='hard,soft')

# new regularization
p.add_argument('--gl_scale', type=int, default=1,help='0:n/a, 1:batchmean, 2:shapemean')
p.add_argument('--gl_scale_coef', type=float, default=100)
p.add_argument('--load', type=str,default='None')


opt = p.parse_args()
if torch.cuda.is_available(): 
     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu 
assert torch.cuda.device_count() == 1

# load configs if exist
if opt.config == '':
     meta_params = vars(opt)
else:
     with open(opt.config,'r') as stream:
          meta_params = yaml.safe_load(stream)
meta_params['single_gpu'] = True
meta_params['use_wandb'] = opt.use_wandb
meta_params['project_title'] = opt.project_title
meta_params['experiment_name'] = opt.experiment_name
meta_params['lr'] = opt.lr
meta_params['loss_grad_deform'] = opt.loss_grad_deform
meta_params['loss_grad_temp'] = opt.loss_grad_temp
meta_params['loss_correct'] = opt.loss_correct
meta_params['pdc_sem'] = opt.pdc_sem
meta_params['pdc_geo'] = opt.pdc_geo
meta_params['gdc_geo'] = opt.gdc_geo
meta_params['assign_latp'] = opt.assign_latp
meta_params['gl_scale'] = opt.gl_scale
meta_params['gl_scale_coef'] = opt.gl_scale_coef
meta_params['load'] = opt.load

# category 
meta_params['category'] = meta_params['point_cloud_path'].split('/')[-2]
######################################################################
print('Training with single gpu')
print('Total subjects: ',meta_params['num_instances'])
print('config: ',opt.config)
if opt.use_wandb:
     wandb.init(project=opt.project_title, name=opt.experiment_name)
     wandb.config.update(meta_params)

# define dataloader
if meta_params['train_split'] == 'None':
     # all_names = sorted(os.listdir(meta_params['point_cloud_path']))
     all_names = os.listdir(meta_params['point_cloud_path'])
     data_path = [os.path.join(meta_params['point_cloud_path'],f) for f in all_names]
else:
     with open(meta_params['train_split'],'r') as file:
          all_names = file.read().split('\n')
     data_path = [os.path.join(meta_params['point_cloud_path'],f + '.mat') for f in all_names]

sdf_dataset = dataset.PointCloudMulti(root_dir=data_path, max_num_instances=meta_params['num_instances'],**meta_params)
dataloader = DataLoader(sdf_dataset, shuffle=True,collate_fn=sdf_dataset.collate_fn,
                    batch_size=meta_params['batch_size'], drop_last = True,num_workers = 8)

##### define Pretrained Encoder #####
model1 = encoder(z_dim=256, num_branch = meta_params['num_branch'])
state_dict = torch.load(meta_params['pretrained_path'])
for key in list(state_dict.keys()):
     if 'pointnet2.' in key:
            state_dict[key.replace('pointnet2.', 'enc_global.')] = state_dict.pop(key)
     elif 'pointnet.' in key:
            state_dict[key.replace('pointnet.', 'enc_global.')] = state_dict.pop(key)
     elif 'generator.' in key:
          state_dict[key.replace('generator.', 'enc_branch.')] = state_dict.pop(key)
model1.load_state_dict(state_dict)

E = model1.enc_global
f = model1.enc_branch
E = E.cuda()
f = f.cuda()
E.eval()
f.eval()

for m in [E, f]:
     for l in m.parameters():
          l.requires_grad = False
model = SAIT(**meta_params)
if meta_params['load'] != 'None':
     model.load_state_dict(torch.load(meta_params['load'],map_location='cpu'))   
model.cuda()

# create save path
root_path = os.path.join(meta_params['logging_root'], meta_params['experiment_name'])
utils.cond_mkdir(root_path)
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
checkpoints_dir = os.path.join(root_path, 'checkpoints')
utils.cond_mkdir(checkpoints_dir)
    
with io.open(os.path.join(root_path,'model.yml'),'w',encoding='utf8') as outfile:
     yaml.dump(meta_params, outfile, default_flow_style=False, allow_unicode=True)

# main training loop
print(meta_params)
training_loop.train(model=model, Encoder=E, ImplicitFun=f, train_dataloader=dataloader, model_dir=root_path, **meta_params)

