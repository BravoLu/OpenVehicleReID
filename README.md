# open-VehicleReID 

## Introduction 

This is a repository for vehicle re-id. Keep updating.

## Installation

### Requirements
* Linux
* CUDA 8.0 or higher 
* Python3
* Pytorch 1.1+

### Install open-VehicleReID
1. Clone the open-VehicleReID repository.
```shell
    https://github.com/BravoLu/open-VehicleReID.git
```

2. Install the dependencies. 
```shell
    cd open-VehicleReID 
    pip -r install requirements.txt 
```

## Get Started

1. Download the dataset ([VeRi776](https://vehiclereid.github.io/VeRi/)\\[VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html)\\[VeRi_Wild](https://github.com/PKU-IMRE/VERI-Wild)).

2. Train
```shell
    python main.py -c {$CONFIG_FILE} \\
                   -d {$DATA_PATH} \\
                   --gpu {$GPU_IDS} \\
                   --seed {$SEED} 
    (e.g.)
    python main.py -c configs/baseline.yml -d /home/share/zhihui/VeRi/ --gpu 0,1 --seed 0 
```

3. Test 
```shell 
    python main.py -c {$CONFIG_FILE} \\
                   -d {$DATA_PATH} \\
                   --ckpt {$CKPT_PATH} \\
                   --gpu {$GPU_IDS} \\
                   --seed {$SEED} \\
                   -t 
    (e.g.)
    python main.py -c configs/baseline.yml -d /home/share/zhihui/VeRi/ --ckpt ckpts/baseline/checkpoint.pth --gpu 0,1 --seed 0
```
## Visualization 


## Benchmark
* VehicleID

|    Model        | Mem (GB) | Rank 1 | Rank 5 | Rank 10 |                                                                                                                 
| :-------------: | :------: | :----: | :----: | :-----: |   
|    Baseline     |    -     |    -   |   -    |    -    |   

* VeRi776

|    Model        | Mem (GB) | mAP | Rank 1 | Rank 5 | Rank 10 |                                                                                                                 
| :-------------: | :-----:  | :-: | :----: | :----: | :-----: | 
|    Baseline     | -        |  -  |    -   |   -    |    -    |  


* VeRi_Wild

|    Model        | Mem (GB) | mAP | Rank 1 | Rank 5 | Rank 10 |                                                                                                                 
| :-------------: | :-----:  | :-: | :----: | :----: | :-----: | 
|    Baseline     | -        |  -  |    -   |   -    |    -    |

  
\* Some code is borrowed from [open-reid](https://github.com/Cysu/open-reid)