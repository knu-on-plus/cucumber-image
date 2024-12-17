#!/bin/bash

# Example bash script to run modal mask generation with different configurations

# python modal_mask_generation.py \
#     --dataset_type train \
#     --position middle \
#     --multi_leaves 0 \
#     --random_ratio False \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/baseline_sample_images" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/baseline_sample_masks" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/baseline_sample_info"

# python modal_mask_generation.py \
#     --dataset_type valid \
#     --position middle \
#     --multi_leaves 0 \
#     --random_ratio False \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/baseline_images_test_valid" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/baseline_masks_test_valid" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/baseline_info_test_valid"

# python modal_mask_generation.py \
#     --dataset_type train \
#     --position random \
#     --multi_leaves 0 \
#     --random_ratio True \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition1_sample_images" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition1_sample_masks" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition1_sample_info"

# python modal_mask_generation.py \
#     --dataset_type valid \
#     --position random \
#     --multi_leaves 0 \
#     --random_ratio True \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition1_images_valid" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition1_masks_valid" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition1_info_valid"

# python modal_mask_generation.py \
#     --dataset_type train \
#     --position random \
#     --multi_leaves 1 \
#     --random_ratio True \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition2_sample_images" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition2_sample_masks" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition2_sample_info"

# python modal_mask_generation.py \
#     --dataset_type valid \
#     --position random \
#     --multi_leaves 1 \
#     --random_ratio True \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition2_images_valid" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition2_masks_valid" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition2_info_valid"

python modal_mask_generation.py \
    --dataset_type train \
    --position random \
    --multi_leaves 2 \
    --random_ratio True \
    --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition3_sample_images" \
    --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition3_sample_masks" \
    --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition3_sample_info"

# python modal_mask_generation.py \
#     --dataset_type valid \
#     --position random \
#     --multi_leaves 2 \
#     --random_ratio True \
#     --save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition3_images_valid_test" \
#     --mask_save_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition3_masks_valid_test" \
#     --json_dir "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/condition3_info_valid_test"
    
