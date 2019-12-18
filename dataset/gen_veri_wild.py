import os
import json
import pickle 

PATH = "/home/shaohao/CVPR20/datas/VeRi_Wild/train_test_split"
IMG_PATH = "/home/shaohao/CVPR20/datas/VeRi_Wild/images/"
raw_train =  "train_list.txt"
raw_test_10000_query = "test_10000_query.txt"
raw_test_10000 = "test_10000.txt"
raw_test_3000 = "test_3000.txt"
raw_test_3000_query =  "test_3000_query.txt"
raw_test_5000 = "test_5000.txt"
raw_test_5000_query = "test_5000_query.txt"
files = [raw_train, raw_test_10000_query, raw_test_10000, raw_test_3000, raw_test_3000_query, raw_test_5000, raw_test_5000_query]
#files = [raw_train]
if __name__ == "__main__":

    newIDs = {}
    with open(os.path.join(PATH, 'sampled_train_1000_id.pkl'), 'rb') as f:
        lines = pickle.load(f)
        #lines = f.readlines()
        for line in lines:
            raw_ID = line
            if raw_ID not in newIDs:
                newIDs[raw_ID] = len(newIDs)

    print(newIDs)

    CamIDs = {}
    TypeIDs = {}
    ModelIDs = {}
    ColorIDs = {}
    IDs = {}

    color2num = {}
    type2num = {}
    model2num = {}

    with open(os.path.join(PATH, "vehicle_info.txt")) as f:
        lines = f.readlines()
        for line in lines:
            fname, camID,_, TypeID, ModelID, ColorID = line.strip().split(';')
            if TypeID in type2num:
                TypeID = type2num[TypeID]
            else:
                type2num[TypeID] = len(type2num)
                TypeID = type2num[TypeID]

            if ModelID in model2num:
                ModelID = model2num[ModelID]
            else:
                model2num[ModelID] = len(model2num)
                ModelID = model2num[ModelID]

            if ColorID in color2num:
                ColorID = color2num[ColorID]
            else:
                color2num[ColorID] = len(color2num) 
                ColorID = color2num[ColorID]
        
            camID = int(camID)
            ID = fname.split('/')[0]
            
            if ID in newIDs:
                IDs[fname] = newIDs[ID]
            else:
                IDs[fname] = ID

            CamIDs[fname] = camID
            ModelIDs[fname] = ModelID
            ColorIDs[fname] = ColorID
            TypeIDs[fname] = TypeID


    for f in files:
        fpath = os.path.join(PATH, f)
        out_file = open('%s.json'%(f.split('.')[0]),"w")
        lines = []
        if f == raw_train:
            with open(os.path.join(PATH, 'sampled_train_1000_id.pkl'), 'rb') as f:
                vids = pickle.load(f)
            for vid in vids:
                #print(vid)
                fnames = os.listdir('images/%s'%vid)
                for fname in fnames:
                    fname = "%s/%s"%(vid, fname.split('.')[0])
                    #print(fname)
                    fpath = os.path.join(IMG_PATH, '%s.jpg'%(fname))
                    ID = IDs[fname]
                    modelID = ModelIDs[fname]
                    colorID = ColorIDs[fname]
                    typeID = TypeIDs[fname]
                    line = {
                        "filename":fname,
                        "vid":ID,
                        "camera":camID,
                        "model":modelID,
                        "color":colorID,
                    }
                    lines.append(line)                    
        else:    
            with open(fpath) as rawf:
                fnames = rawf.readlines()
                for fname in fnames:
                    fname = fname.strip()
                    fpath = os.path.join(IMG_PATH, fname+'.jpg')
                    ID = IDs[fname]
                    camID = CamIDs[fname]
                    modelID = ModelIDs[fname]
                    colorID = ColorIDs[fname]
                    typeID = TypeIDs[fname]
                    line = {
                        "fllename":fname,
                        "vid":ID,
                        "camera":camID,
                        "model":modelID,
                        "color":colorID,
                    }
                    lines.append(line)
                #out_file.write("{} {} {} {} {} {} {}\n".format(fpath, ID, camID, modelID, colorID, typeID, fname))
        print(len(lines))
        json.dump(lines, out_file, indent=4, ensure_ascii=False)


    
    


