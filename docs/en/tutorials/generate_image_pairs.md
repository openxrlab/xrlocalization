# Generate image pairs

## For reconstruction

**Step 0**
```commandline
python3 tools/ir_create_database.py \
    --image_dir /path/to/image_dirctory \
    --database_path /path/to/database.bin
```

**Step 1**
```commandline
python3 tools/ir_image_retrieve.py \
    --database_path /path/to/database.bin \
    --save_path /path/to/save/pair.txt \
    --retrieve_num 20
```
