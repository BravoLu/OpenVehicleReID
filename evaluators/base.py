import torch 
import torch.nn as nn
from torch.autograd import Variable
import os
import copy
import random
from collections import OrderedDict, defaultdict
import numpy as np
from sklearn.metrics import average_precision_score
import pickle 
import pdb 

from utils import *

class BaseEvaluator(object):
    def __init__(self, cfg, model, dataset):

        self.cfg = cfg 
        self.model = model 
        if isinstance(self.model, nn.DataParallel):
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.num_gpus = torch.cuda.device_count() 

            if self.num_gpus > 1:
                self.model = nn.DataParallel(model)
            
            self.model.to(self.device)

        _, self.test_loader, self.query, self.gallery, self.vids = dataset 

    def evaluate(self):

        self.model.eval()
        features, labels = self.extract_features()
        
        if self.cfg['TARGET'] == 'VehicleID':
            rank1s, rank5s, rank10s = [], [], []
            self.test = self.query + self.gallery  
            for _ in range(10):
                self.query = self.test.copy() 
                self.gallery = []
                random.shuffle(self.query)
                for vid in self.vids:
                    for item in self.query:
                        if item[1] == vid:
                            self.gallery.append(item)
                            self.query.remove(item)
                            break 

                distmat = self.pairwise_distance(features)
                _, CMC = self.calculate_mAP_CMC(distmat)
                rank1s.append(CMC[0])
                rank5s.append(CMC[4])
                rank10s.append(CMC[9])

            rank1 = np.array(rank1s).mean()
            rank5 = np.array(rank5s).mean()
            rank10 = np.array(rank10s).mean()

            return rank1, rank5, rank10 
        else:
            distmat = self.pairwise_distance(features)
            mAP, CMC = self.calculate_mAP_CMC(distmat)

            return mAP, CMC

    
    def rank_list_visualization(self, save_file):
        
        query_names = np.asarray([q[-1] for q in self.query])
        gallery_names = np.asarray([g[-1] for g in self.gallery])
        query_cams = np.asarray([q[2] for q in self.query])
        gallery_cams = np.asarray([g[2] for g in self.gallery])

        features, labels = self.extract_features()
        distmat = self.pairwise_distance(features)
        indices = np.argsort(distmat.numpy(), axis=1)
        m, _ = distmat.shape
        rank_list = []
        for i in range(m):
            num = 0 
            rank_list_i = []
            valid = (gallery_cams[indices[i]] != query_cams[i])
            for idx, j in enumerate(indices[i]):
                if valid[idx]:
                    rank_list_i.append(str(gallery_names[j]))
                    num += 1
                if num > 35:
                    break 
            rank_list.append(rank_list_i)

        with open(save_file, 'wb') as f:
            pickle.dump(rank_list, f)
        
    def extract_features(self):
        '''
            Return:
                features: dicts of {'fname': feature}
                labels: dicts of {'fname': label}
        '''
        features = OrderedDict()
        labels = OrderedDict()

        bar = Bar('Extracting features', max=len(self.test_loader))
        for i,inputs in enumerate(self.test_loader):
            imgs, vids, fnames = Variable(inputs[0]).to(self.device), inputs[1].to(self.device), inputs[-1]
            self.model.eval()
            feats = self.model(imgs)
            feats = feats.data.cpu()
            for fname, feat, vid in zip(fnames, feats, vids):
                features[fname] = feat
                labels[fname] = vid.item()  
                bar.suffix = '[{cur}/{amount}]'.format(cur=i+1, amount=len(self.test_loader))
            bar.next()
        bar.finish()

        return features, labels 

    def pairwise_distance(self, features):
        '''
        x = torch.cat([features[q[-1]].unsqueeze(0) for q in self.query], dim=0)
        y = torch.cat([features[g[-1]].unsqueeze(0) for g in self.gallery], dim=0)
        '''
        #pdb.set_trace()
        x = torch.stack([features[q[-1]] for q in self.query])
        y = torch.stack([features[g[-1]] for g in self.gallery])

        if self.cfg['RERANKING']:
            dist = re_ranking(x, y, 20, 6, 0.3)
        else:
            m,n = x.size(0), y.size(0)
            x = x.view(m, -1)
            y = y.view(n,-1)
            dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m,n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,m).t()
            dist.addmm_(1, -2, x, y.t())

        return dist

    def calculate_mAP_CMC(self, distmat):
        query_ids = [q[1] for q in self.query]
        gallery_ids = [g[1] for g in self.gallery]
        query_cams = [q[2] for q in self.query]
        gallery_cams = [g[2] for g in self.gallery]
        
        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        if self.cfg['TARGET'] == 'VehicleID':
            mAP = None
        else:
            mAP = self.mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
        CMC_scores = self.CMC(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

        return mAP, CMC_scores

    def mean_ap(self, distmat, query_ids, gallery_ids, query_cams, gallery_cams):
        distmat = distmat.cpu().numpy()
        m, _ = distmat.shape
        
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
        
        aps = []
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) | 
                    (gallery_cams[indices[i]] != query_cams[i]))
            
            y_true = matches[i, valid]
            y_score = -distmat[i][indices[i]][valid]

            if not np.any(y_true): continue
            aps.append(average_precision_score(y_true, y_score))
        if len(aps) == 0:
            raise RuntimeError("No valid query")
        return np.mean(aps)

    def CMC(self, distmat, query_ids, gallery_ids, query_cams, gallery_cams, topk=100):
        distmat = distmat.cpu().numpy()
        m, _ = distmat.shape
        
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_ids[indices] == query_ids[:, np.newaxis])

        ret = np.zeros(topk)
        num_valid_queries = 0
        for i in range(m):
            valid = ((gallery_ids[indices[i]] != query_ids[i]) | 
                    (gallery_cams[indices[i]] != query_cams[i]))

            y_true = matches[i, valid]
            if not np.any(y_true): continue
            index = np.nonzero(y_true)[0]
            if index.flatten()[0] < topk:
                ret[index.flatten()[0]] += 1
            num_valid_queries += 1
        if num_valid_queries == 0:
            raise RuntimeError("No valid query")
        
        return ret.cumsum() / num_valid_queries