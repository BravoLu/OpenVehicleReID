from torch.utils.data import Dataset
from PIL import Image 
import os 
import json 

class BaseDataset(object):
    def __init__(self, root=''):
        self.train, self.test, self.gallery, self.query = [], [], [], [] 
        self.root = root 
        self.load() 

    def load(self):
        raise NotImplementedError 

    def _load_json(self, json_file):
        with open(json_file, 'r') as f:
            dicts = json.load(f)
        return dicts
        
    def __repr__(self):       
        str = "  set          | # images  \n" \
              "  train        | # %s      \n" \
              "  test         | # %s      \n" \
              "  gallery      | # %s      \n" \
              "  query        | # %s      \n" \
              "  train format | # %s      \n" \
              "  test  format | # %s      \n"%(len(self.train), len(self.test), len(self.gallery), len(self.query), self.train[0], self.test[0])
        
        return str 

