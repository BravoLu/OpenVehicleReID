from __future__ import absolute_import 
import os
import errno
import yaml
import torch 
import os.path as osp 
import shutil 

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def parse_config(args):
    
    
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg['DATA_ROOT'] = args.data
    cfg['SEED'] = args.seed 
    cfg['GPU_NUMS'] = len(args.gpu.split(','))
    cfg['LOG_DIR'] = args.log
    cfg['CKPT'] = args.ckpt
    cfg['TEST'] = args.test
    cfg['RESUME'] = args.resume
    cfg['CKPT_DIR'] = args.save
    cfg['RERANK'] = args.rerank
    cfg['use_apex'] = args.use_apex
    return cfg


def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

        
    
    