# %%
import utils
import os
import pathlib
import argparse
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
import torch 
import mymodels 
import mydataset 
from torch.utils.data import DataLoader
from utils.myfed import *
import yaml
# %%
if __name__ == "__main__":
    yamlfilepath = pathlib.Path(__file__).parent.absolute().joinpath('config.yaml')
    args = yaml.load(yamlfilepath.open('r'), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    handlers = [logging.StreamHandler()]
    args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
    
    writer = SummaryWriter(comment=args.logfile, comet_config={'disabled': False})
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if args.debug:
        writer = None
        handlers.append(logging.FileHandler(
            f'./logs/debug.txt', mode='a'))
    else:
        handlers.append(logging.FileHandler(
            f'./logs/{args.logfile}.txt', mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers,     
    )
    logging.info(args)
    
    # 1. data
    args.datapath = os.path.expanduser(args.datapath)
    
    if args.dataset == 'cifar10':
        publicdata = 'cifar100'
        args.N_class = 10
    elif args.dataset == 'cifar100':
        publicdata = 'imagenet'
        args.N_class = 100
    elif args.dataset == 'pascal_voc2012':
        publicdata = 'mscoco'
        args.N_class = 20
    
    assert args.dataset in ['cifar10', 'cifar100', 'pascal_voc2012']
    
    priv_data, _, test_dataset, public_dataset, distill_loader = mydataset.data_cifar.dirichlet_datasplit(
        args, privtype=args.dataset, publictype=publicdata, N_parties=args.N_parties, online=not args.oneshot, public_percent=args.public_percent)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, sampler=None)
    
    ###########
    # 2. model
    logging.info("CREATE MODELS.......")
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    logging.info(f'GPU: {args.gpu}')
    # model = resnet8(num_classes=args.N_class).cuda()
    model = mymodels.define_model(modelname=args.model_name, num_classes=args.N_class, pretrained=args.pretrained)

    logging.info("totally {} paramerters".format(np.sum(x.numel() for x in model.parameters())))
    logging.info("Param size {}".format(np.sum([np.prod(x.size()) for name,x in model.named_parameters() if 'linear2' not in name])))
    
    # 3. fed training
    fed = FedMAD(model, distill_loader, priv_data, test_loader, writer, args)
    if args.oneshot:
        fed.update_distill_loader_wlocals(public_dataset)
        fed.distill_local_central_oneshot()
    else:
        fed.distill_local_central()
    
    if not args.debug:
        writer.close()


