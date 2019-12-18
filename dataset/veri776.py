from .base import BaseDataset
import json 
import os.path as osp

class VeRi776(BaseDataset):
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
    def load(self):
        for dict in self._load_json('dataset/veri776_train.json'):
            self.train.append([osp.join(self.root, 'image_train', dict['filename']),
                               dict['vid'],
                               dict['camera'],
                               dict['model'],
                               dict['color']
                            ])

        for dict in self._load_json('dataset/veri776_query.json'):
            self.query.append([osp.join(self.root, 'image_query', dict['filename']),
                               dict['vid'], 
                               dict['camera'],
                               dict['model'], 
                               dict['color'],
                               dict['filename'].split('.')[0],
                            ])

        for dict in self._load_json('dataset/veri776_gallery.json'):
            self.gallery.append([osp.join(self.root, 'image_test', dict['filename']),
                                 dict['vid'],
                                 dict['camera'],
                                 dict['model'],
                                 dict['color'],
                                 dict['filename'].split('.')[0],
                            ])
        
        self.test = self.gallery + self.query 

'''
d = VeRi776(root='/home/share/zhihui/VeRi')
print(d)
'''