import argparse
import os 
import random 
import numpy as np 
import torch 

from utils import * 
from dataset import *
from dataset.market1501 import Market1501
from models import *
from trainers import *
from evaluators import *

def parse_args():
    parser = argparse.ArgumentParser('Unsupervised Vehicle ReID')
    parser.add_argument('-c', '--config', type=str,
                                  help='the path to the training config', default='baseline')
    parser.add_argument('-t', '--test', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, help='the path of checkpoint file.')
    parser.add_argument('--save', type=str, help='the save path of checkpoint file.')
    parser.add_argument('-s', '--check', action='store_true', default=False, help="for fast check complie error in program")
    parser.add_argument('-d', '--data', type=str, default='/home/share/zhihui/VeRi', help='the path of dataset')
    parser.add_argument('-v', '--vis', type=str, default=None, help='the path of the output file of rank list visualization')
    parser.add_argument('--log', type=str, default='default', help='the path of the log file.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=str, default="", help='the path of resume checkpoint.')
    parser.add_argument('--rerank', action='store_true')
    args = parser.parse_args()
    print(args)
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args, cfg):
    
    dataset = get_dataloader(configs, root=args.data, quick_check=args.check)
    print('dataset ids', DATASET_ID_NUM[cfg['TARGET']])
    model = globals()[cfg['MODEL']](DATASET_ID_NUM[cfg['TARGET']])
    trainer = globals()[cfg['TRAINER']](cfg, model, dataset)
    trainer.train()

def test(args, cfg):

    dataset = get_dataloader(configs, root=args.data, quick_check=False)
    model = globals()[cfg['MODEL']](DATASET_ID_NUM[cfg['TARGET']])
    evaluator = globals()[cfg['EVALUATOR']](cfg, model, dataset)
    if cfg['TARGET'] == 'VehicleID':
        rank1, rank5, rank10 = evaluator.evaluate() 
    else:
        mAP, CMC = evaluator.evaluate() 
        rank1, rank5, rank10 = CMC[0], CMC[4], CMC[9]

    if cfg['TARGET'] != 'VehicleID':
        print('mAP: %.5f'%mAP)
    print('RANK-1: %.5f\nRANK-5: %.5f\nRANK-10: %.5f'%(rank1, rank5, rank10))
 
    if args.vis != None:
        evaluator.rank_list_visualization(args.vis)
    

if __name__ == "__main__":
    
    args = parse_args()
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
 
    configs = parse_config(args)
    if args.test:
        test(args, configs)
    else:
        train(args, configs)
