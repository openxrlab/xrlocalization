# Quick Start
XRLoc provides a flexible tool that can easily perform visual localization offline and online.
Given a query image, XRLoc estimates a 6DoF pose from a pre-reconstructed map. A tiny dataset
is provided for convenience. Donwload the dataset from 
[here](https://sensear-map.oss-cn-hangzhou.aliyuncs.com/stloc_sdk/xrloc-test-data.tar.gz). The 
folder extracted from the dataset is shown below.
```commandline
├── map
│   ├── database.bin
│   ├── features.bin
│   ├── images.bin
│   └── points3D.bin
├── query
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── 000003.jpg
│   ├── 000004.jpg
│   ├── 000005.jpg
│   └── 000006.jpg
└── query.txt
```
Here, the pre-reconstructed map contains the sparse models(images.bin and points3D.bin), 3D point 
descriptors(features.bin), and image global feature(database.bin). The query folder contains six 
query images. The file(query.txt) records camera intrinsic for every query image. 

## Run localization offline
```commandline
python3 run_benchmark.py --map_path /path/to/map --query_path /path/to/query.txt
```
This would take several minutes to automatically download models at first time.
The tool does not store any results. If you intend to use the estimated camera poses, 
the log produced by the tool should be stored in a log file.
```commandline
python3 run_benchmark.py --map_path /path/to/map --query_path /path/to/query.txt > result.log &
```
Then, you can use the provided log parsing script to extract camera pose from the log file.
```commandline
python3 tools/xrloc_log_parser.py --logs /path/to/result.log
```

## Run localization online