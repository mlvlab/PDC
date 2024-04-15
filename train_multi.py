import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import yaml
import configargparse
import wandb
import io

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import dataset, utils, training_loop
from model import SAIT, encoder
from utils import is_main_process


def main_worker(gpu, ngpus_per_node, port, config): #  main,
    # cudnn.benchmark = True
    print(f'Use GPU: {gpu}')
    config['gpu'] = gpu
    config['world_size'] = ngpus_per_node
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '%d' % (port)
    dist.init_process_group(backend='nccl',init_method='env://', world_size=ngpus_per_node, rank=gpu)
    main(config)
    

def deploy_model(config):
    ### Pretrained Encoder ###
    if 'human' in config['category']:
        model1 = encoder(z_dim=256, num_branch = config['num_branch'],L1_dim = 1024, L2_dim = 128)
    else:
        model1 = encoder(z_dim=256, num_branch = config['num_branch'])
    state_dict = torch.load(config['pretrained_path'])
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
    torch.cuda.set_device(config['gpu'])
    E.cuda(config['gpu'])
    f.cuda(config['gpu'])
    E = torch.nn.parallel.DistributedDataParallel(E, device_ids=[config['gpu']], find_unused_parameters=True)
    f = torch.nn.parallel.DistributedDataParallel(f, device_ids=[config['gpu']], find_unused_parameters=True)
    
    model = SAIT(**config)
    if config['load'] != 'None':
        model.load_state_dict(torch.load(config['load'],map_location='cpu'))   

    model.cuda()
    model.cuda(config['gpu'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['gpu']], find_unused_parameters=True)

    return E,f,model
    

def main(config):
    # if config['gpu'] == 0:
    if is_main_process():
        if config['use_wandb']:
            wandb.init(project=config['project_title'],name=config['experiment_name'])
            wandb.config.update(config)
    # define dataloader
    if config['train_split'] == 'None':
        all_names = sorted(os.listdir(config['point_cloud_path']))
        # all_names = all_names[:8]
        data_path = [os.path.join(config['point_cloud_path'],f) for f in all_names]
    else:
        with open(config['train_split'],'r') as file:
            all_names = file.read().split('\n')
        data_path = [os.path.join(config['point_cloud_path'],f + '.mat') for f in all_names]

    sdf_dataset = dataset.PointCloudMulti(root_dir=data_path, max_num_instances=config['num_instances'],**config)
    if torch.cuda.device_count() != 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(sdf_dataset,shuffle=True,num_replicas=config['world_size'],rank=config['gpu']) # 
        
        mp_context = torch.multiprocessing.get_context('fork')
        dataloader = DataLoader(sdf_dataset, collate_fn=sdf_dataset.collate_fn,sampler=train_sampler,
                        batch_size=config['batch_size'], drop_last = True,num_workers = 8,multiprocessing_context=mp_context)
    # model
    E,f,model = deploy_model(config)
    E.eval()
    f.eval()
    
    for m in [E, f]:
        for l in m.parameters():
            l.requires_grad = False
    
    # # create save path
    root_path = os.path.join(config['logging_root'], config['experiment_name'])
    if is_main_process():
        with io.open(os.path.join(root_path,'model.yml'),'w',encoding='utf8') as outfile:
            yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)

    # main training loop
    training_loop.train(model=model, Encoder=E, ImplicitFun=f, train_dataloader=dataloader, model_dir=root_path, **config)


if __name__ == '__main__':
    ## Arguments ##
    p = configargparse.ArgumentParser()
    
    p.add_argument('--use_wandb', type=bool, default=False)
    p.add_argument('--project_title', type=str, default='XXX')
    p.add_argument('--config', type=str, default='configs/train/XXX.yml')
    p.add_argument('--experiment_name', type=str, default='XXX') 
    
    #### tuning parameters ###################################################
    p.add_argument('--lr', type=float, default=0.0001)

    # regarding deformation
    p.add_argument('--loss_grad_deform', type=float,default=10, help='deformation smoothness prior.')
    p.add_argument('--loss_grad_temp', type=float,default=100, help='loss weight for normal consistency prior.')
    p.add_argument('--loss_correct', type=float,default=100)
    p.add_argument('--pdc_sem', type=float, default=50)
    p.add_argument('--pdc_geo', type=float, default=750)
    p.add_argument('--gdc_geo', type=float, default=100)
    ###########################################################################
    p.add_argument('--assign_latp',type=str,default='soft',help='hard,soft')


    # new regularization
    p.add_argument('--gl_scale', type=int, default=1,help='0:n/a, 1:batchmean, 2:shapemean')
    p.add_argument('--gl_scale_coef', type=float, default=100)
    p.add_argument('--load', type=str,default='None')
    
    opt = p.parse_args()
    with open(opt.config,'r') as stream:
        meta_params = yaml.safe_load(stream)
    meta_params['single_gpu'] = False
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
    meta_params['category'] = meta_params['point_cloud_path'].split('/')[3]
    ######################################################################
    print('Training with multiple gpu')
    print('Total subjects: ',meta_params['num_instances'])
    print('config: ',opt.config)
    
    world_size = torch.cuda.device_count()
    
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, port, meta_params))
