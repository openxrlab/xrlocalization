# Use your own data

If you want to collect your own scene data, you can use the [acquisition tool]() 
to capture data. Then we recommend using [XRSfM]() or [COLMAP](https://colmap.github.io/)
to generate the initial sparse reconstruction. Next, 
please refer to [here](../tutorials/generate_loc_map.md) to generate the 
localization map.

## Offline Evaluation
If you want to evaluate visual localization offline,  please refer to 
[benchmark evaluation](../benchmark/benchmark_evaluation.md).


## Online Test
We provide an [APP]() that can request localization online, and show the AR 
effect in combination with visual localization and [SLAM]().