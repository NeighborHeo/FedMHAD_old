# %%
import sys
sys.path.append('/home/suncheol/code/FedTest/FedMAD')
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

yamlfilepath = pathlib.Path(__file__).parent.parent.absolute().joinpath('config.yaml')
with yamlfilepath.open('r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
args = argparse.Namespace(**config)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=config.get("model_name"), help="model name")
parser.add_argument("--sublossmode", default=config.get("sublossmode"), help="at or mha")
parser.add_argument("--task", default=config.get("task"), help="task name")
parser.add_argument("--distill_heads", default=config.get("distill_heads"), help="distill heads")
parser.add_argument("--lambda_kd", default=config.get("lambda_kd"), help="lambda kd")
args = parser.parse_args(namespace=args)

print (args)
print (args.initepochs)