from .base import BaseDataset
import json 
import os.path as osp 

class VeRi_Wild(BaseDataset):
    '''
    Label format (e.g.):
    [
        {
            'filename': str,
            'vid': int,
            'camera': str, 
            'model': int,
            'color': int, 
        }
    ]
    '''
    def __init__(self, root, test_size='large'):
        assert test_size in ['small', 'median', 'large']
        self.query_json = 'dataset/veriwild_query_%s.json'%test_size
        self.gallery_json = 'dataset/veriwild_gallery_%s.json'%test_size
        self.train, self.test, self.gallery, self.query = [], [], [], []
        self.root = root 
        self.load()

    def load(self):
        for dict in self._load_json('dataset/veriwild_train.json'):
            self.train.append([osp.join(self.root, 'images', dict['filename']),
                               dict['vid'],
                               dict['camera'],
                               dict['filename'].split('.')[0],
                            ])

        for dict in self._load_json(self.query_json):
            self.query.append([osp.join(self.root, 'images', dict['filename']),
                               dict['vid'], 
                               dict['camera'],
                               dict['filename'].split('.')[0],
                            ])

        for dict in self._load_json(self.gallery_json):
            self.gallery.append([osp.join(self.root, 'images', dict['filename']),
                                 dict['vid'],
                                 dict['camera'],
                                 dict['filename'].split('.')[0],
                            ])
        
        self.test = self.gallery + self.query 


'''
d = VeRi_Wild(root='/home/share/zhihui/VeRi/')
'''
