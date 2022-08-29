# Retriangulate a spare model with a new feature
In this part, we introduce how to re-triangulate a sparse SfM model with a new feature. For example,
assume there is a sparse SfM model reconstructed with SIFT feature, we intend to use a deep feature, 
such as D2Net or SuperPoint, to do visual localization. 

Assume we have reference images and the sparse model that is COLMAP format:
```commandline
+---images/
|    |-- ...
+---model/
     |--cameras.bin
     |--images.bin
     |--points3D.bin
```
This would take a few steps.

**0.** Extract local feature
```commandline
python3 tools/recon_feature_extract.py \
       --image_dir /path/to/images \
       --image_bin_path /path/to/model/images.bin \
       --feature_bin_path /path/to/features.bin \
       --extractor d2net
```
This would extract d2net feature for every image in `/path/to/images` and save as `features.bin`.

**1.** Perform 2D-2D feature matching
```commandline
python3 tools/recon_feature_extract.py \
       --recon_path /path/to/model \
       --feature_bin_path /path/to/features.bin \
       --match_bin_path /path/to/matches.bin 
```
This would produce a file `matches.bin` that records the matching info about all image pairs.

**2.** Perform triangulation
Please refer to [XRSfM](https://github.com/openxrlab/xrsfm) for how to do triangulation.
This would produce the new sparse model stored at `/path/to/new_model`