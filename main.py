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

# import comet_ml at the top of your file
from comet_ml import Experiment

torch.autograd.set_detect_anomaly(True)

# %%
if __name__ == "__main__":
    # set seed 
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
     
    yamlfilepath = pathlib.Path(__file__).parent.absolute().joinpath('config.yaml')
    with yamlfilepath.open('r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace(**config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=config.get("model_name"), help="model name")
    parser.add_argument("--sublossmode", default=config.get("sublossmode"), help="at or mha")
    parser.add_argument("--task", default=config.get("task"), help="task name")
    parser.add_argument("--distill_heads", type=int, default=config.get("distill_heads"), help="distill heads")
    parser.add_argument("--lambda_kd", type=float, default=config.get("lambda_kd"), help="lambda kd")
    args = parser.parse_args(namespace=args)

    # args to dict 
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="3JenmgUXXmWcKcoRk8Yra0XcD",
        project_name=f"fedmhad-{args.task}",
        workspace="neighborheo",
    )
    experiment.log_parameters(vars(args))
    
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    handlers = [logging.StreamHandler()]
    args.logfile = f'{datetime.now().strftime("%m%d%H%M")}'+args.logfile
    
    writer = SummaryWriter(comment=args.logfile) # comet_config={'disabled': False})
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
    fed = FedMAD(central= model, distil_loader= distill_loader, private_data= priv_data, val_loader= test_loader, writer=None, experiment=experiment, args=args)
    # if args.oneshot:
    #     fed.update_distill_loader_wlocals(public_dataset)
    #     fed.distill_local_central_oneshot()
    # else:
    fed.distill_local_central()
    print('done')
    
    if not args.debug:
        writer.close()


