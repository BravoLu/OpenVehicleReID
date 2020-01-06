import time 
import os 

import torch.nn as nn
import torch 
from torch.autograd import Variable 
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np 

from evaluators import *
from utils import *
from loss import * 

class BaseTrainer(object):
    def __init__(self, cfg, model, dataset):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg 
        self.logger = Logger(cfg)
        self.train_loader, self.test_loader, self.query, self.gallery, _ = dataset 

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() 
        self.model = model 
        
        if self.num_gpus >= 1:
            self.model = nn.DataParallel(model)
        
        self.model.to(self.device)
        
        self.pretrained = self.cfg['PRETRAINED']

        # resume
        self.resume = self.cfg['RESUME']
        if os.path.exists(self.resume):
            infos = torch.load(self.resume)
            model.load_state_dict(infos['state_dict'])
            self.start_epoch = infos['epoch']
            self.best_rank1 = infos['best_rank1']
            self.logger.write(' loaded\n'%self.resume)
        else:
            self.start_epoch = 0
            self.best_rank1 = 0

        # pretrained 
        if os.path.exists(self.pretrained):
            model.load_param(self.pretrained)      

        self.optimizer = make_optimizer(self.cfg, self.model, num_gpus=self.num_gpus)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg['STEPS'])
        #self.scheduler = WarmupMultiStepLR(self.optimizer, self.cfg['STEPS'], self.cfg['GAMMA'], self.cfg['WARMUP_FACTOR'])

        self.evaluator = globals()[cfg['EVALUATOR']](self.cfg, self.model, dataset)

    def train(self):
        triplet = TripletLoss().cuda()
        ce = nn.CrossEntropyLoss().cuda()

        for epoch in range(self.start_epoch, self.cfg['EPOCHS']):
            self.model.train() 
            self.scheduler.step()
            stats = {'id_loss', 'triplet_loss', 'total_loss'}
            meters_trn = {stat: AverageMeter() for stat in stats}

            bar = Bar('Epoch[{N_epoch}]'.format(N_epoch=epoch), max=len(self.train_loader))
            start_time = time.time()
            
            for i,inputs in enumerate(self.train_loader):
                imgs = Variable(inputs[0]).to(self.device)
                labels = Variable(inputs[1]).to(self.device)

                cls_scores, feats = self.model(imgs)

                id_loss = ce(cls_scores, labels)
                triplet_loss = triplet(feats, labels)

                total_loss = self.cfg['ALPHA'] * id_loss + self.cfg['BETA'] * triplet_loss 

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step() 

                for k in stats:
                    v = locals()[k]
                    meters_trn[k].update(v.item(), self.cfg['BATCH'])

                bar.suffix = '[{N_batch}/{N_size}] | Loss:{N_loss:.3f} {N_lossa:.3f} |'.format(
                    N_epoch=epoch, 
                    N_batch=i+1, 
                    N_size=len(self.train_loader), 
                    N_loss=meters_trn['total_loss'].val, 
                    N_lossa=meters_trn['total_loss'].avg
                    )
                bar.next()

            self.logger.write('\n\nTraining Infos: \n')
            self.logger.write('[%d] \nlr              : %.7f \n'%(epoch+1, self.scheduler.get_lr()[0]))
            for loss in stats:
                self.logger.write('%s : %.2f \n'%(loss.ljust(15), meters_trn[loss].avg))
            self.logger.write('Time consumed   : %.1fs\n'%(time.time() - start_time))

            bar.finish()
            if epoch % self.cfg['EVAL_FRE'] == 0:
                self.evaluate(epoch, stats)

    def evaluate(self, epoch, stats=None):

        if self.cfg['TARGET'] == 'VehicleID':
            rank1, rank5, rank10 = self.evaluator.evaluate()
        else:
            mAP, cmc = self.evaluator.evaluate()
            rank1, rank5, rank10 = cmc[0], cmc[4], cmc[9]
        
        is_best = rank1 > self.best_rank1
        self.best_rank1 = max(rank1, self.best_rank1)
        
        self.logger.write('Evaluation: \n')
        if self.cfg['TARGET'] != 'VehicleID':
            self.logger.write("mAP: {:3.1%}\n".format(mAP))
        self.logger.write('RANK-1: {:3.1%}\n'.format(rank1))
        self.logger.write('RANK-5: {:3.1%}\n'.format(rank5))
        self.logger.write('RANK-10: {:3.1%}\n'.format(rank10))

        mkdir_if_missing(self.cfg['CKPT_DIR'])
        if self.num_gpus >= 1:
            save_checkpoint({
                'state_dict':self.model.module.state_dict(),
                'epoch':epoch+1,
                'best_rank1': self.best_rank1,
            }, is_best=is_best, fpath=os.path.join(self.cfg['CKPT_DIR'], 'checkpoint.pth'))
        else:
            save_checkpoint({
                'state_dict':self.model.state_dict(),
                'epoch':epoch+1,
                'best_rank1': self.best_rank1,
            }, is_best=is_best, fpath=os.path.join(self.cfg['CKPT_DIR'], 'checkpoint.pth'))

        
