# Generate localization map
XRLocalization map includes four files: `images.bin`, `points3D.bin`, `features.bin` and `database.bin`.
In this part, we introduce how to generate this map.

The format of `images.bin` and `points3D.bin` is the same as COLMAP format. `features.bin` records
3D descriptor for all 3D point in `points3D.bin`. Every 3D descriptor is represented as the mean of
all it corresponding 2D local features.  `database.bin` records image global features for all image
recorded in `images.bin`.

The prerequisite is re-triangulation. Two type of re-triangulation results are supported.
* Re-triangulation by [xrloc](retriangulation.md), requiring prerequisites including:
`images.bin`, `points3D.bin` and `features.bin`
* Re-triangulation by [hloc](https://github.com/cvg/Hierarchical-Localization),
requiring prerequisites including: `images.bin`, `points3D.bin` and `feats-xxxx.h5`

**Step 0** Generate `images.bin` `point3Ds.bin` `features.bin`

xrloc:
```commandline
python3 tools/loc_convert_reconstruction.py \
    --feature_path /path/to/features.bin \
    --model_path /path/to/include/images.bin/and/points3Ds.bin/directory \
    --output_path /path/to/map/directory
```
This would produce three new file `images.bin`, `points3D.bin`, `features.bin`
in the path `/path/to/map/directory`.

hloc:
```commandline
python3 tools/loc_convert_reconstruction.py \
    --feature_path /path/to/feats-xxxx.h5 \
    --model_path /path/to/include/images.bin/and/points3Ds.bin/directory \
    --output_path /path/to/map/directory
```
The only difference is at `--feature_path /path/to/feats-xxxx.h5`.  The output is the same as xrloc.


**Step 1** Generate `database.bin`
```commandline
python3 tools/ir_create_database.py \
   --image_dir /path/to/image_dir \
   --image_bin_path /path/to/step0/map/directory/images.bin \
   --databse_path /data/to/database.bin \
   --extractor netvlad
```
This would extract `netvlad` feature for all images in `image_dir` and
save as `database.bin`.


Finally, merge the output from two steps as localization map.
