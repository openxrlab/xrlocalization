# Quick Start
XRLocalization provides a flexible tool that can easily perform visual localization offline and online.
Given a query image, XRLocalization estimates a 6DoF pose from a pre-reconstructed map. A tiny dataset
is provided for convenience. Download the dataset from
[here](https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrlocalization/meta/xrloc-test-meta.tar.gz). The
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
│   └── 000003.jpg
└── query.txt
```
Here, the pre-reconstructed map contains the sparse models (`images.bin` and `points3D.bin`), 3D point
descriptors (`features.bin`), and image global feature database (`database.bin`). The query folder
contains three query images. The file `query.txt` records camera intrinsic for every query image.

## Run localization offline
```commandline
python3 run_benchmark.py --map_path /path/to/map --query_path /path/to/query.txt
```
This would take several minutes to automatically download models at first time.
Please refer to [here](benchmark/benchmark_evaluation.md) for more details on benchmark evaluation
using this tool.



## Run localization online
```commandline
python3 run_web_server.py --map_path /path/to/map --port 12345
```
This command would start a localization server, which listen to
the client's localization request and return the corresponding
localization result. Please refer to [here](https://github.com/openxrlab/xrdocument) for details on how to
use this server at the client.
