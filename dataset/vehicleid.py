from .base import BaseDataset
import json 
import os.path as osp 
import random 
 
class VehicleID(BaseDataset):
    '''
    (For consistency, we assume the filename as its corresponding
        camera, because this dataset do not have camera id.)
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

    def __init__(self, root="/home/share/zhihui/VehicleID_V1.0/", test_size='small'):
        assert test_size in ['small', 'median', 'large']
        self.test_json = "dataset/vehicleid_test_%s.json"%test_size 
        self.train, self.test, self.gallery, self.query = [], [], [], []
        self.root = root 
        self.load()

    def load(self):
        '''
        The VehicleID dataset is a little bit different from other dataset (e.g. VeRi776, VeRi-Wild). It randomly takes one image per vehicle
        ID to form the gallery, and the rest of images to form the query set.
        According the original papper "http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Deep_Relative_Distance_CVPR_2016_paper.pdf"
        '''
        for dict in self._load_json('dataset/vehicleid_train.json'):
            self.train.append([osp.join(self.root, 'image', dict['filename']),
                               dict['vid'],
                               dict['camera'],
                               dict['model'],
                               dict['color'],
                            ])
        
        self.vids = set()
        for dict in self._load_json(self.test_json):
            self.test.append([osp.join(self.root, 'image', dict['filename']),
                              dict['vid'],
                              dict['camera'],
                              dict['filename'].split('.')[0],
                            ])
            self.vids.add(dict['vid'])
        '''
            We random shuffle the test set, then sample the first image each vehicle ID as the gallery
        '''
        self.query = self.test.copy()
        self.gallery = [] 

        random.shuffle(self.query)
        for vid in self.vids:
            for item in self.query:
                if item[1] == vid:
                    self.gallery.append(item)
                    self.query.remove(item)
                    break 


if __name__ == "__main__":
    d = VehicleID(test_size='large')
    print(d)

