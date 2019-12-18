import os 
import json 

with open('/home/share/zhihui/VehicleID_V1.0/train_test_split/test_800.txt', 'r') as f:
	lines = f.readlines()
lists = []
for line in lines:
	line = line.strip() 
	filename, vid = line.split(' ')
	dict = {'filename':filename, 'vid':int(vid), 'camera':filename.split('.')[0]}
	lists.append(dict)
with open('vehicleid_test_small.json', 'w') as f:
	json.dump(lists, f, indent=4, ensure_ascii=False)


with open('/home/share/zhihui/VehicleID_V1.0/train_test_split/test_1600.txt', 'r') as f:
	lines = f.readlines() 
lists = []
for line in lines:
	line = line.strip() 
	filename, vid = line.split(' ')
	dict = {'filename':filename, 'vid':int(vid), 'camera':filename.split('.')[0]}
	lists.append(dict)
with open('vehicleid_test_median.json', 'w') as f:
	json.dump(lists, f, indent=4, ensure_ascii=False)


with open('/home/share/zhihui/VehicleID_V1.0/train_test_split/test_2400.txt', 'r') as f:
	lines = f.readlines()
lists = []
for line in lines:
	line = line.strip() 
	filename, vid = line.split(' ')
	dict = {'filename':filename, 'vid':int(vid), 'camera':filename.split('.')[0]}
	lists.append(dict)
with open('vehicleid_test_large.json', 'w') as f:
	json.dump(lists, f, indent=4, ensure_ascii=False)
