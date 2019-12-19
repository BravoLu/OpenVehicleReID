# This is a visualization tool of vehicle re-identification.
# Step 1: Save the matched test image IDs for each query image as a text file, where each line contains a test image ID ranked in terms of distance score in ascending order. Name each text file as '%06d.txt' % <query_image_ID>. We assume that the top-35 matched test images are displayed. An example is given in "./dist_example/".
# Step 2: Run "python visualize.py".
# Step 3: Input the path of the directory containing all text files at "Txt Dir:" (end with '/'). An example is given as "./dist_example/".
# Step 4: Click "Load".
# The query image is shown on the top left. The corresponding test images are shown on the right.
# For each image, the image ID is shown on the top left corner. 
# Click "<< Prev" to return to the previous query.
# Click "Next >>" to advance to the next query.
# Enter the query no. and click "Go" to jump to the corresponding query.  

from __future__ import division
from __future__ import print_function
from tkinter import *
#import tkMessageBox

import argparse 
import pickle
from PIL import Image, ImageTk
import os
import glob
import cv2
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser('Rank List')
parser.add_argument('-d', '--data', type=str)
args = parser.parse_args()

DATA_PATH = args.data

# image sizes for the examples
SIZE = 256, 256

def load_pickle(path):
    with open(path, 'rb') as f:
        dicts = pickle.load(f)
    return dicts

def num_per_id():
    imgs = os.listdir('../image_test')
    gallery_num = defaultdict(int)
    for img in imgs:
        ID = img[0:4]
        gallery_num[ID] += 1

    return gallery_num

gallery_num = num_per_id()

class VisTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("VisTool")
        self.frame = Frame(self.parent)
        self.frame['background'] = 'white'
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.txtDir = ''
        self.txtList = []
        self.prbList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = ''
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text="Rank File(.pkl):")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry = Entry(self.frame)
        #self.info = Label(self.frame, text="info")
        #self.info.grid(row=1, column=1, sticky=W + E)
        self.entry.grid(row=0, column=1, sticky=W + E)
        self.ldBtn = Button(self.frame, text="Load", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)

        # main panel
        self.mainPanel = Canvas(self.frame, cursor='arrow')
        self.parent.bind("a", self.prevPrb)  # press 'a' to go backforward
        self.parent.bind("d", self.nextPrb)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevPrb)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextPrb)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Query No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoPrb)
        self.goBtn.pack(side=LEFT)

        # panel for query image
        self.prbPanel = Frame(self.frame, border=10)
        self.prbPanel['background'] = 'white'
        self.prbPanel.grid(row=1, column=0, rowspan=5, sticky=N)
        self.tmpLabel2 = Label(self.prbPanel, text="Query image:")
        self.info = Label(self.prbPanel, text="info:")
        self.info.pack(side=BOTTOM, pady=10)
        self.info['background'] = 'white'
        self.tmpLabel2['background'] = 'white'
        self.tmpLabel2.pack(side=TOP, pady=5)
        self.prbLabels = []
        self.prbLabels.append(Label(self.prbPanel))
        self.prbLabels[-1].pack(side=TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

    def loadDir(self):
        #self.txtDir = os.path.join(self.entry.get())
        self.rank_pkl = load_pickle(self.entry.get()+'.pkl')
        self.queries = list(self.rank_pkl.keys())
        #self.txtList = glob.glob(os.path.join(self.txtDir, '*.txt'))
        #self.txtList.sort(key=lambda x: int(x[-10:-4]))

        if len(self.queries) == 0:
            print('No text file found in the specified directory!')
            return

        # default: the 1st image in the collection
        self.cur = 1
        self.total = len(self.queries)

        self.loadImage()
        print('%d images loaded' % self.total)

    def loadImage(self):

        query = self.queries[self.cur - 1]

        rank_list = []

        rank_list.append(os.path.join(DATA_PATH, 'image_query', query))

        for gallery in self.rank_pkl[query]:
            rank_list.append(os.path.join(DATA_PATH, 'image_test', gallery))

        self.tmp = []
        self.prbList = []

        # load query image
        f = rank_list[0]
        print(f)
        gn = gallery_num[query[0:4]]
        self.info.config(text="gallery size: %d"%(gn))
        im = cv2.imread(f)
        #cv2.imwrite('test.jpg', im)
        query_id = f[-24:-20]
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im).resize((200, 200), Image.BILINEAR)
        #im = Image.fromarray(im).resize((200, 200), Image.ANTIALIAS)

        im = np.asarray(im) #cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
        cv2.putText(im, f[-24:-4], (20,40), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_h, im_w = im.shape[:2]
        if (im_w + im_h) < 100:
            thk = 2
        else:
            thk = int((im_w + im_h) / 50)
        # cv2.rectangle(im, (0, 0), ((im_w-1), (im_h-1)), (0,0,255), thk)
        im = Image.fromarray(im)
        #im = im.convert('RBG')
        r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
        new_size = int(r * im.size[0]), int(r * im.size[1])
        self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
        self.prbList.append(ImageTk.PhotoImage(self.tmp[-1]))
        self.prbLabels[0].config(image=self.prbList[-1], width=SIZE[0], height=SIZE[1])

        # load gallery images
        rank_list = rank_list[1:]
        siz = len(rank_list)
        siz = min(siz, 35)
        myrange = np.array(range(siz))
        myrange = myrange + 35 * 0
        myimage = []
        for i in myrange:
            print(rank_list[i])
            im = cv2.imread(rank_list[i])
            im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            im = Image.fromarray(im).resize((200, 200), Image.BILINEAR)
            im = np.asarray(im)
            gallery_id = rank_list[i][-24:-20]
            if query_id == gallery_id:
                cv2.rectangle(im,(2,2),(im.shape[1]-2,im.shape[0]-2),(0,255,0),2)
            else:
                cv2.rectangle(im,(2,2),(im.shape[1]-2,im.shape[0]-2),(255,0,0),2)            
            cv2.putText(im, rank_list[i][-24:-4], (5,40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0),1)

            im_h, im_w = im.shape[:2]
            if (im_w + im_h) < 100:
                thk = 2
            else:
                thk = int((im_w + im_h) / 50)

            # if p_id == g_id:
                # cv2.rectangle(im, (0, 0), ((im_w-1), (im_h-1)), (0,255,0), thk)
            # else:
                # cv2.rectangle(im, (0, 0), ((im_w-1), (im_h-1)), (255,0,0), thk)
            '''
            if im.shape[0] > im.shape[1]:
                myimage.append(Image.fromarray(im).resize((int(200 * im.shape[1] / im.shape[0]), 200), Image.ANTIALIAS))
            else:
                myimage.append(Image.fromarray(im).resize((200, int(200 * im.shape[0] / im.shape[1])), Image.ANTIALIAS))
            '''
            myimage.append(Image.fromarray(im))
        self.img = Image.new('RGB', (1400, 180 * 5))

        ss = 7
        for i in range(siz):
            row = int(i / ss)
            tlx = (i - ss * row) * 200
            tly = 180 * row
            shape = myimage[i].size
            self.img.paste(myimage[i], (tlx, tly, tlx + shape[0], tly + shape[1]))

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 1000), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

    def prevPrb(self, event=None):
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextPrb(self, event=None):
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoPrb(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.cur = idx
            self.loadImage()


if __name__ == '__main__':
    root = Tk()
    tool = VisTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()
