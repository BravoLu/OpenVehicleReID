import os
import os.path as osp
from xml.dom.minidom import parse
import xml.dom.minidom
import json
from tqdm import tqdm 
from collections import defaultdict
ROOT='/home/share/zhihui/VeRi'
TRAIN_PATH='veri776_train.json'
GALLERY_PATH='veri776_gallery.json'
QUERY_PATH='veri776_query.json'



def parse_train_xml():
    DOMTree = xml.dom.minidom.parse('train_label.xml')
    Items = DOMTree.documentElement
    item_list = Items.getElementsByTagName("Item")
    origin_ID = []
    new_ID = {}
    for item in item_list:
        origin_ID.append(item.getAttribute("vehicleID"))

    origin_ID = sorted(set(origin_ID))
    for new,old in enumerate(origin_ID):
        new_ID[old] = new

    lines = []
    for item in tqdm(item_list):
        imgName = item.getAttribute('imageName')
        fname = osp.splitext(imgName)[0]
        line = { 
            "filename":imgName,
            "vid":int(new_ID[item.getAttribute('vehicleID')]),
            "camera":item.getAttribute('cameraID'),
            "color":int(item.getAttribute('colorID')),
            "model":int(item.getAttribute('typeID')),
        }
        lines.append(line)
        with open(TRAIN_PATH, 'w') as f:
            json.dump(lines, f, indent=4, ensure_ascii=False)


def parse_test_xml():
    tmp_set = open(osp.join(ROOT,'name_query.txt'),'r').readlines()
    query_set = [ item.strip() for item in tmp_set]
    DOMTree = xml.dom.minidom.parse('test_label.xml')
    Items = DOMTree.documentElement
    item_list = Items.getElementsByTagName("Item")
    query_lines, gallery_lines = [],[] 
    for item in tqdm(item_list):
        imgName = item.getAttribute('imageName')
        fname = osp.splitext(imgName)[0]
        line = {
            "filename":imgName,
            "vid":int(item.getAttribute('vehicleID')),
            "camera":item.getAttribute('cameraID'),
            "color":int(item.getAttribute('colorID')),
            "model":int(item.getAttribute('typeID')),
        }
        if imgName in query_set:
            query_lines.append(line)
        else:
            gallery_lines.append(line)

    with open(QUERY_PATH, 'w') as f:
        json.dump(query_lines, f, indent=4, ensure_ascii=False)
    with open(GALLERY_PATH, 'w') as f:
        json.dump(gallery_lines, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    parse_train_xml()
    parse_test_xml()
