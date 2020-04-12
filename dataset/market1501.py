from .base import BaseDataset
import json 
import os.path as osp

class Market1501(BaseDataset):
    def load(self):
        for dict in self._load_json('dataset/Market1501_trainval.json'):
            self.train.append([
                                dict['filename'],
                                dict['pid'],
                                dict['cam'],
                                osp.basename(dict['filename']).split('.')[0]
                            ])

        for dict in self._load_json('dataset/Market1501_query.json'):
            self.query.append([
                                dict['filename'],
                                dict['pid'],
                                dict['cam'],
                                osp.basename(dict['filename']).split('.')[0]
                            ])

        for dict in self._load_json('dataset/Market1501_gallery.json'):
            self.gallery.append([
                                dict['filename'],
                                dict['pid'],
                                dict['cam'],
                                osp.basename(dict['filename']).split('.')[0]
                            ])
        
        self.test = self.gallery + self.query 
