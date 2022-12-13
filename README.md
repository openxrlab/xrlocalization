# XRLocalization

<div align="left">

[![actions](https://github.com/openxrlab/xrlocalization/workflows/build/badge.svg)](https://github.com/openxrlab/xrlocalization/actions)
[![codecov](https://codecov.io/gh/openxrlab/xrlocalization/branch/main/graph/badge.svg)](https://codecov.io/gh/openxrlab/xrlocalization)
[![LICENSE](https://img.shields.io/github/license/openxrlab/xrlocalization.svg)](https://github.com/openxrlab/xrlocalization/blob/main/LICENCE)

</div>


## Introduction
English | [简体中文](README_CN.md)

XRLocalization is an open source visual localization toolbox based on Python. It is a
part of the OpenXRLab project.

https://user-images.githubusercontent.com/111835288/187731838-45ccbfaf-8a19-46a1-bd53-29c56793642c.mp4

### Major Features

* Robust and efficient large-scale feature-based visual localization
* Both offline and online visual localization are supported
* A hierarchical framework that can easily integrate new features and matching methods

## Getting Started
* [Installation](docs/en/installation.md)
* [Quick Start](docs/en/get_started.md)
* [Benchmark Evaluation](docs/en/benchmark/benchmark_evaluation.md)
* [Customizing New Feature](docs/en/tutorials/customize_local_feature.md)
* [Customizing Your Map](docs/en/tutorials/customize_map.md)
* [Building AR Application](http://139.196.36.90/openxrlab_document/ARDemo/ARdemo.html)

## Benchmark
Please refer to [benchmark](docs/en/benchmark/benchmark.md).

## AR Demo
Please refer to [here](http://139.196.36.90/openxrlab_document/ARDemo/ARdemo.html) for building your own AR application.
[Here](https://user-images.githubusercontent.com/44204704/187864126-e9cd7a43-a773-487d-ad01-4cc2988f3b5a.mp4) 
is an AR demo based on XRLocalization.

## License
The license of our codebase is [Apache-2.0](LICENSE). Note that this license only 
applies to code in our library, the dependencies of which are separate and individually 
licensed. We would like to pay tribute to open-source implementations to which we rely 
on. Please be aware that using the content of dependencies may affect the license of 
our codebase. Some supported methods may 
carry [additional licenses](docs/en/additional_licenses.md).

## FAQ
Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Citation
If you use this toolbox or benchmark in your research, please cite this project.
```bibtex
@misc{xrlocalization,
    title={OpenXRLab Visual Localization Toolbox and Server},
    author={XRLocalization Contributors},
    howpublished = {\url{https://github.com/openxrlab/xrlocalization}},
    year={2022}
}
```
If you use Geometry-Aided Matching in your research, please cite:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2211.08712,
  url = {https://arxiv.org/abs/2211.08712},
  author = {Yu, Hailin and Feng, Youji and Ye, Weicai and Jiang, Mingxuan and Bao, Hujun and Zhang, Guofeng},
  title = {Improving Feature-based Visual Localization by Geometry-Aided Matching},
  publisher = {arXiv},
  year = {2022},
}
```
and
```bibtex
@inproceedings{yu2020learning,
  title={Learning bipartite graph matching for robust visual localization},
  author={Yu, Hailin and Ye, Weicai and Feng, Youji and Bao, Hujun and Zhang, Guofeng},
  booktitle={2020 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  pages={146--155},
  year={2020},
  organization={IEEE}
}
```

## Contributing
We appreciate all contributions to improve XRLocalization.
Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement
XRLocalization is an open source project that is contributed by researchers and
engineers from both the academia and the industry.
We appreciate all the contributors who implement their methods or add new features,
as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community
by providing a flexible toolkit to reimplement existing methods and develop their
own new models.


## Projects in OpenXRLab
- [XRPrimer](https://github.com/openxrlab/xrprimer): OpenXRLab foundational library for XR-related algorithms.
- [XRSLAM](https://github.com/openxrlab/xrslam): OpenXRLab Visual-inertial SLAM Toolbox and Benchmark.
- [XRSfM](https://github.com/openxrlab/xrsfm): OpenXRLab Structure-from-Motion Toolbox and Benchmark.
- [XRLocalization](https://github.com/openxrlab/xrlocalization): OpenXRLab Visual Localization Toolbox and Server.
- [XRMoCap](https://github.com/openxrlab/xrmocap): OpenXRLab Multi-view Motion Capture Toolbox and Benchmark.
- [XRMoGen](https://github.com/openxrlab/xrmogen): OpenXRLab Human Motion Generation Toolbox and Benchmark.
- [XRNeRF](https://github.com/openxrlab/xrnerf): OpenXRLab Neural Radiance Field (NeRF) Toolbox and Benchmark.
