import os
from PIL import Image

class Preprocessor(object):
    def __init__(self, dataset, training=True, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.training = training
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        '''
        if self.training:
            imgpath, vid, _, model, color = self.dataset[index]
            img = Image.open(imgpath).convert('RGB')
            
            img = self.transform(img)

            fname = os.path.basename(imgpath).split('.')[0]

            return img, vid, model, color, fname
        else:
            imgpath, vid, camera, fname = self.dataset[index] 
            img = Image.open(imgpath).convert('RGB')
            img = self.transform(img) 

            #fname = os.path.basename(imgpath).split('.')[0]
            return img, vid, camera, fname 
        '''
        imgpath, id_, cam, fname = self.dataset[index]
        img = Image.open(imgpath).convert('RGB')
        img = self.transform(img)
        
        return img, id_, cam, fname 