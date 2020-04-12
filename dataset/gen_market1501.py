import os 
import re 
from glob import glob 
import json 

def register(absdir, subdir, output, pattern = re.compile(r'([-\d]+)_c(\d)'), reset=True):
    fpaths = sorted(glob(os.path.join(absdir, subdir, '*.jpg')))
    items = [] 
    pid_sets = {}
    for fpath in fpaths:
        fname = os.path.basename(fpath)
        pid, cam = map(int, pattern.search(fname).groups())
        if pid == -1: continue 
        if reset:
            if pid not in pid_sets:
                pid_sets[pid] = len(pid_sets)
            pid = pid_sets[pid]
        assert 0 <= pid <= 1501
        assert 1 <= cam <= 6 
        cam -= 1 
        line = {
            'filename': fpath, 
            'pid': pid,
            'cam': cam,
        }
        items.append(line)
    
    with open(output, 'w') as f:
        json.dump(items, f, indent=4, ensure_ascii=False)

register('/home/shaohao/data/Market-1501-v15.09.15', 'bounding_box_train', 'Market1501_trainval.json', reset=True)
register('/home/shaohao/data/Market-1501-v15.09.15', 'bounding_box_test', 'Market1501_gallery.json', reset=False)
register('/home/shaohao/data/Market-1501-v15.09.15', 'query', 'Market1501_query.json', reset=False)
