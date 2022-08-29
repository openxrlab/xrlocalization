# Benchmark Evaluation
In this part, we introduce how to do evaluation on public datasets, which would 
provide reference images and query images. The reference image has usually been 
reconstructed using SfM and provides the corresponding camera poses. We provide
a tool `run_benchmark.py` for evaluation. 
```commandline
python3 run_benchmark.py \
    --map_path /path/to/localization_map \
    --query_path /path/to/query_file \
    --json /path/to/json_file
```

## Preparation

Before running `run_benchmark.py`, we need to provide `localization map`, `query file` 
and a `json file`.

**0.** Generate `localization map`

Please refer to [generate_loc_map.md](../tutorials/generate_loc_map.md).


**1.** Construct `query file`

The tool `run_benchmark.py` requires a file that records the query image info. Each query
image occupied one line, which has the format:
```commandline
query_image_name camera_model_name image_width image_height intrinsic_param_list
```
* `query_image_name`: Relative path to image
* `camera_model_name`: Support all COLMAP [camera model](https://colmap.github.io/cameras.html)
* `image_widht`: The width of the corresponding query image
* `image_height`: The height of the corresponding query image
* `intrinsic_param_list`: Camera intrinsic params that is depended on `camera_model_name`.
For example, if `camera_model_name` is set to `PINHOLE`, `intrinsic_param_list` has the
format `fx fy cx cy`.

**2.** Set configuration through `json_file`

We can specify configuration through a `json_file`. We provide a template json at 
`configs/sample.json`.
```json
{
    "local_feature": "d2net",
    "global_feature": "netvlad",

    "matcher": "gam",
    "coarse": "sr",

    "retrieval_num": 20,
    "scene_size": 20,

    "max_inlier": 50,
    "max_scene_num": 2
}
```
Note that both `local_feature` and `global_feature` must be the same as when constructing
localization map.

## Extracting results
The tool does not store any results. If you intend to use the estimated camera poses, 
the log produced by the tool should be stored in a log file. For example, 
```commandline
python3 run_benchmark.py \
    --map_path /path/to/map \ 
    --query_path /path/to/query.txt \
    --json /path/to/json_file > result.log &
```
Then, you can use the provided log parsing script to extract camera pose from the log file 
and do you want to do.
```commandline
python3 tools/loc_log_parser.py --logs /path/to/result.log
```