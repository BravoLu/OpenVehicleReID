import torchvision.transforms as T 
from torch.utils.data import DataLoader
import torch
from torch.utils.data import dataset, sampler
import math 
import random 
from collections import defaultdict
import copy 
import numpy as np

from .vehicleid import VehicleID
from .veri776 import VeRi776
from .veri_wild import VeRi_Wild 
from .preprocessor import Preprocessor

class RandomErasing(object):
    def __init__(self,probability=0.5,sl=0.02,sh=0.4,r1=0.3,mean=(0.4914,0.4822,0.4465)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
    def __call__(self, img):
        if random.uniform(0,1) >= self.probability:
            return img
        for _ in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset : VehicleID/VeRi776/VeRi_Wild
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, dataset, batch_size, num_instances):
        self.data_source = dataset.train
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        #changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

def get_dataloader(cfg, root ,quick_check=False):

    target = globals()[cfg['TARGET']](root=root)
    print(target)
    if quick_check:
        target.train[:] = target.train[:1000]
    
    #num_gpus = torch.cuda.device_count() 
    if cfg['TARGET'] == 'VehicleID':
        vids = target.vids
    else:
        vids = None 

    normalizer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transformer = T.Compose([
        T.Resize((cfg['WIDTH'], cfg['HEIGHT'])),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(0),
        T.RandomCrop((cfg['WIDTH'],cfg['HEIGHT'])),
        T.ToTensor(),
        normalizer,
        RandomErasing(),
    ])
    
    test_transformer = T.Compose([
        T.Resize((cfg['WIDTH'], cfg['HEIGHT'])),
        T.ToTensor(),
        normalizer,
    ])
    
    train_loader = DataLoader(
        Preprocessor(target.train, training=True, transform=train_transformer),
        batch_size=cfg['BATCH'],
        sampler=RandomIdentitySampler(target, cfg['BATCH'], cfg['INSTANCE']),
        num_workers=4, 
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        Preprocessor(target.test, training=False, transform=test_transformer),
        batch_size=cfg['BATCH'],
        shuffle=False,
        num_workers=4,
        pin_memory=True, 
    )

    return train_loader, test_loader, target.query, target.gallery, vids