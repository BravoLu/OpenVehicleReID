from torch.utils.data import Dataset
from PIL import Image 
import os 
import json 

class BaseDataset(object):
    def __init__(self, root=''):
        self.train, self.test, self.gallery, self.query = [], [], [], [] 
        self.root = root 
        self.load() 
        self.training = True 
    
    def load(self):
        raise NotImplementedError 
    
    def set_transformer(self, train_transformer, test_transformer):
        self.train_transformer = train_transformer
        self.test_transformer = test_transformer

    def _load_json(self, json_file):
        with open(json_file, 'r') as f:
            dicts = json.load(f)
        return dicts
        
    def __len__(self):
        if self.training:
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, index):
        if self.training:
            imgpath, vid, _, model, color = self.train[index]
            img = Image.open(imgpath).convert('RGB')
            
            img = self.train_transformer(img)

            fname = os.path.basename(imgpath).split('.')[0]

            return img, vid, model, color, fname
        else:
            imgpath, vid, camera, _, _, fname = self.test[index] 
            img = Image.open(imgpath).convert('RGB')
            img = self.test_transformer(img) 

            #fname = os.path.basename(imgpath).split('.')[0]
            return img, vid, camera, fname 

    def __repr__(self):       
        str = "  set          | # images  \n" \
              "  train        | # %s      \n" \
              "  test         | # %s      \n" \
              "  gallery      | # %s      \n" \
              "  query        | # %s      \n" \
              "  train format | # %s      \n" \
              "  test  format | # %s      \n"%(len(self.train), len(self.test), len(self.gallery), len(self.query), self.train[0], self.test[0])
        
        return str 

