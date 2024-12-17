import json
import os, argparse
import cv2
import numpy as np
import random
from tqdm import tqdm
from custom_utils import *
from amodal_utils import *
from coco_json import initialize_coco_json, save_coco_json, add_to_coco_json
from visualization_utils import visualize_merged_amodal_and_modal, visualize_all_masks
from config import INPUT_PATHS, OUTPUT_PATHS, HYPERPARAMETERS

def merge_leaf_to_cucumber(cucumber_image, leaf_image, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio):
    # 오이 객체 중심 계산
    cucumber_bbox = get_bbox_from_mask(cucumber_mask)
    # 위치에 따른 좌표 계산
    leaf_location = calculate_leaf_location(cucumber_bbox, position)
    #print(f'leaf_location: {leaf_location}')
    leaf_image = leaf_size_initialization(cucumber_mask, leaf_image, initial_leaf_ratio)
    
    #print(f'position: {position}, occlusion_ratio: {occlusion_ratio}')
    resized_leaf_image = resize_leaf_to_target_ratio(cucumber_mask, leaf_image, leaf_location, occlusion_ratio)
    
    merged_image, leaf_mask = merge_and_crop_leaf(cucumber_image, resized_leaf_image, leaf_location)

    return merged_image, leaf_mask

def merge_multi_leaves_to_cucumber(cucumber_image, leaf_image1, leaf_image2, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio):
    # 오이 객체 중심 계산
    cucumber_bbox = get_bbox_from_mask(cucumber_mask)
    # 위치에 따른 좌표 계산
    leaf_location1 = calculate_leaf_location(cucumber_bbox, 'top')
    leaf_location2 = calculate_leaf_location(cucumber_bbox, 'bottom')

    #print(f'occlusion_ratio: {occlusion_ratio}')
    leaf_image1 = leaf_size_initialization(cucumber_mask, leaf_image1, initial_leaf_ratio)
    leaf_image2 = leaf_size_initialization(cucumber_mask, leaf_image2, initial_leaf_ratio)

    leaf_location1, leaf_location2 = adjust_leaves_to_occlusion(cucumber_mask, leaf_image1, leaf_image2, leaf_location1, leaf_location2, occlusion_ratio)
    merged_image1, leaf_mask1 = merge_and_crop_leaf(cucumber_image, leaf_image1, leaf_location1)
    merged_image2, leaf_mask2 = merge_and_crop_leaf(merged_image1, leaf_image2, leaf_location2)
    final_leaf_mask = cv2.bitwise_or(leaf_mask1, leaf_mask2)
    return merged_image2, final_leaf_mask

def synthesize_images(cucumber_image_path, cucumber_mask_path, leaf_image_paths, position, occlusion_ratio, initial_leaf_ratio, 
                      save_dir=None, global_image_id=0, target_size=(768, 1024), multi_leaves=0):
    # 오이 이미지와 잎 이미지 로드
    cucumber_image = cv2.imread(cucumber_image_path, cv2.IMREAD_UNCHANGED)
    leaf_image = cv2.imread(leaf_image_paths[0], cv2.IMREAD_UNCHANGED)
    cucumber_mask = cv2.imread(cucumber_mask_path, cv2.IMREAD_GRAYSCALE)
    # 이미지를 합성
    
    if multi_leaves in [1,2]:
        #print(f"Processing leaf image: {os.path.basename(leaf_image_paths[0]), os.path.basename(leaf_image_paths[1])}")
        leaf_image2 = cv2.imread(leaf_image_paths[1], cv2.IMREAD_UNCHANGED)
        if multi_leaves == 1:
            merged_image, leaf_mask = merge_multi_leaves_to_cucumber(cucumber_image, leaf_image, leaf_image2, cucumber_mask, position, occlusion_ratio,
                                                                 initial_leaf_ratio)
        else:
            overlapped_leaves = overlap_dual_leaves(cucumber_mask, leaf_image, leaf_image2, initial_leaf_ratio)
            merged_image, leaf_mask = merge_leaf_to_cucumber(cucumber_image, overlapped_leaves, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio)

    else:    
        #print(f"Processing leaf image: {os.path.basename(leaf_image_paths[0])}")
        merged_image, leaf_mask = merge_leaf_to_cucumber(cucumber_image, leaf_image, cucumber_mask, position, occlusion_ratio, initial_leaf_ratio)

    # 리사이즈
    resized_image, resized_masks = resize_image_and_masks(
        merged_image, [cucumber_mask, leaf_mask], target_size=target_size)
    amodal_mask, leaf_mask = resized_masks
    
    # 합성 이미지 저장
    cucumber_image_name = os.path.basename(cucumber_image_path)
    merged_image_name = f"{os.path.splitext(cucumber_image_name)[0]}_merged_{global_image_id:06d}_{occlusion_ratio}.png"
    resized_image_path = save_image(save_dir, merged_image_name, resized_image)
    
    return resized_image_path, amodal_mask, leaf_mask

def generate_coco_annotation(coco_json, amodal_mask, modal_mask, leaf_mask, global_image_id, global_annotation_id, merged_image_path):
    #print("COCO Format 데이터 생성 시작...")
    # 5. 공통 Annotation 생성
    cucumber_annotation = generate_annotation(
        amodal_mask=amodal_mask,
        modal_mask = modal_mask,
        global_id=global_annotation_id,
        image_id=global_image_id,
        category_id=1,
        occluder_segm=mask_to_polygon(leaf_mask)
    )
    coco_json["annotations"].append(cucumber_annotation)
    global_annotation_id += 1

    # 6. 잎 Annotation 생성
    leaf_annotation = generate_annotation(
        amodal_mask=leaf_mask,
        modal_mask = None,
        global_id=global_annotation_id,
        image_id=global_image_id,
        category_id=2,  # 잎 클래스 ID
    )
    coco_json["annotations"].append(leaf_annotation)
    global_annotation_id += 1
    
    # 7. 이미지 정보 추가
    image_info = {
        "id": global_image_id,
        "width": int(amodal_mask.shape[1]),
        "height": int(amodal_mask.shape[0]),
        "file_name": os.path.basename(merged_image_path),
    }
    coco_json["images"].append(image_info)

    return coco_json, global_annotation_id

def process_amodal_images_and_masks(cucumber_image_path, leaf_cropped_image_paths, cucumber_mask_path, save_dir, mask_save_dir, 
                                    coco_json, global_image_id, global_annotation_id, position, occlusion_ratio, initial_leaf_ratio, 
                                    multi_leaves=0):
    
    # 오이 이미지에 잎 이미지를 합성하고 저장
    #print("오이 이미지 합성 시작...")
    merged_image_path, amodal_mask, leaf_mask = synthesize_images(cucumber_image_path, cucumber_mask_path, leaf_cropped_image_paths, 
                                                                    position, occlusion_ratio, initial_leaf_ratio, save_dir, global_image_id, 
                                                                    multi_leaves=multi_leaves)
    #print("Modal 마스크 생성 시작...")
    # 3. Modal 마스크 생성 및 겹치는 부분 (가림) 정보 계산
    modal_mask, overlap_mask = get_amodal_masks(amodal_mask, leaf_mask)
    
    save_processed_masks(amodal_mask, overlap_mask, modal_mask, os.path.basename(merged_image_path), mask_save_dir)
    
    #print("COCO Format 데이터 생성 시작...")
    # 4. COCO Format 데이터 생성
    coco_json, global_annotation_id = generate_coco_annotation(
        coco_json, amodal_mask, modal_mask, leaf_mask, global_image_id, global_annotation_id, merged_image_path)

    # ID 증가
    global_image_id += 1

    # 9. 시각화
    #visualize_merged_amodal_and_modal(cv2.imread(merged_image_path, cv2.IMREAD_UNCHANGED), amodal_mask, modal_mask)

    return coco_json, global_image_id, global_annotation_id

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic configuration for mask generation.")
    parser.add_argument("--dataset_type", type=str, choices=["train", "valid", "debugging"], default="debugging",
                        help="Dataset type to process.")
    parser.add_argument("--position", type=str, choices=["top", "middle", "bottom", "random"], default="random",
                        help="Position of the leaf on the cucumber.")
    parser.add_argument("--multi_leaves", type=int, choices=[0, 1, 2], default=0,
                        help="Whether to use multiple leaves. 0: single leaf, 1: dual leaves, 2: overlap dual leaves.")
    parser.add_argument("--random_ratio", type=bool, default=False,
                        help="Whether to use random occlusion ratio.")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Path to save synthesized images.")
    parser.add_argument("--mask_save_dir", type=str, required=True,
                        help="Path to save modal masks.")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Path to save COCO-style JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    from config import INPUT_PATHS, OUTPUT_PATHS, HYPERPARAMETERS

    args = parse_arguments()

    # Update HYPERPARAMETERS dynamically
    HYPERPARAMETERS["dataset_type"] = args.dataset_type
    HYPERPARAMETERS["position"] = args.position
    HYPERPARAMETERS["multi_leaves"] = args.multi_leaves
    HYPERPARAMETERS["random_ratio"] = args.random_ratio

    # Update OUTPUT_PATHS dynamically
    OUTPUT_PATHS["save_dir"] = args.save_dir
    OUTPUT_PATHS["mask_save_dir"] = args.mask_save_dir
    OUTPUT_PATHS["json_dir"] = args.json_dir

    # Rest of the main script logic here
    print(f"Updated configuration:\nDataset Type: {HYPERPARAMETERS['dataset_type']}\n"
          f"Position: {HYPERPARAMETERS['position']}\nMulti-leaves: {HYPERPARAMETERS['multi_leaves']}\n"
          f"Random Ratio: {HYPERPARAMETERS['random_ratio']}\n"
          f"Save Dir: {OUTPUT_PATHS['save_dir']}\nMask Save Dir: {OUTPUT_PATHS['mask_save_dir']}\n"
          f"JSON Dir: {OUTPUT_PATHS['json_dir']}")

    ''' input'''
    dataset_type = HYPERPARAMETERS["dataset_type"]
    cucumber_images_dir = INPUT_PATHS["cucumber_images_dir"] + dataset_type
    cucumber_masks_dir = INPUT_PATHS["cucumber_masks_dir"] + dataset_type
    leaf_cropped_dir = INPUT_PATHS["leaf_cropped_dir"] + dataset_type

    ''' output dir'''
    save_dir = OUTPUT_PATHS["save_dir"]
    mask_save_dir = OUTPUT_PATHS["mask_save_dir"]
    json_dir = OUTPUT_PATHS["json_dir"]

    '''paramters '''

    image_index_start = HYPERPARAMETERS["image_index_start"]
    sample_count = 0   # 현재 생성된 샘플 수
    position = HYPERPARAMETERS["position"]  # 잎이 오이를 어디에 위치할지
    multi_leaves = HYPERPARAMETERS["multi_leaves"]  # 한 잎 이미지에 대해 여러 오이 이미지에 합성할지 여부

    random_ratio = HYPERPARAMETERS["random_ratio"]  # 랜덤
    ratios = HYPERPARAMETERS["r_settings"]  # 랜덤
    proportions = HYPERPARAMETERS["r_proportions"]  # 랜덤

    initial_leaf_ratio = HYPERPARAMETERS["initial_leaf_ratio"]  # 잎 이미지 초기 비율
    sort = HYPERPARAMETERS["sort"]  # 이미지 정렬

    '''초기화'''
    ensure_directories_exist([save_dir, mask_save_dir, json_dir])

    sample_limit = 2000 if dataset_type == "train" else 5 # train = 10000, val = 4000

    if random_ratio:
        occlusion_ratio_list = create_occlusion_ratio_list(sample_limit, ratios, proportions)
    else:
        occlusion_ratio = HYPERPARAMETERS["occlusion_ratio"]  # 잎이 오이를 얼마나 가리는지 비율


    # 특정 클래스 마스크만 선택 (클래스 0: 오이)
    def get_cucumber_masks(mask_dir, image_name):
        cucumber_masks = []
        for mask_file in os.listdir(mask_dir):
            # 파일명이 이미지명과 매칭되고 클래스가 0인 마스크만 선택
            if mask_file.startswith(image_name) and '_0_' in mask_file:
                cucumber_masks.append(os.path.join(mask_dir, mask_file))
        return cucumber_masks


    # cucumber 이미지 파일 불러오기
    cucumber_image_paths = get_image_paths_from_folder(cucumber_images_dir, sort=sort)

    # COCO JSON 초기화
    coco_json = initialize_coco_json()

    from tqdm import tqdm

    # 유효한 오이 이미지와 마스크 필터링
    valid_cucumber_paths = []
    for cucumber_image_path in cucumber_image_paths:
        image_name = os.path.splitext(os.path.basename(cucumber_image_path))[0]
        cucumber_mask_paths = get_cucumber_masks(cucumber_masks_dir, image_name)
        if len(cucumber_mask_paths) > 0:
            valid_cucumber_paths.append((cucumber_image_path, cucumber_mask_paths))

    # 유효한 오이 이미지 수 계산
    total_cucumber_images = len(valid_cucumber_paths)
    if total_cucumber_images == 0:
        raise ValueError("유효한 오이 마스크가 있는 이미지가 없습니다!")

    # 샘플 수 계산
    sample_per_cucumber = sample_limit // total_cucumber_images
    remaining_samples = sample_limit % total_cucumber_images

    print(f"유효한 오이 이미지 수: {total_cucumber_images}")
    print(f"오이 이미지 당 샘플 수: {sample_per_cucumber}, 추가 샘플 수: {remaining_samples}")

    # tqdm로 진행 상태 표시
    with tqdm(total=sample_limit, desc="Generated samples", unit="samples") as pbar:
        sample_count = 0
        global_image_id, global_annotation_id = 0, 0

        # 각 cucumber 이미지에 대해 마스크와 잎 합성
        for cucumber_idx, (cucumber_image_path, cucumber_mask_paths) in enumerate(valid_cucumber_paths):
            # 오이 이미지 당 생성할 샘플 수
            samples_for_this_cucumber = sample_per_cucumber
            if cucumber_idx < remaining_samples:
                samples_for_this_cucumber += 1

            # 각 오이 이미지에서 생성된 샘플 수 추적
            cucumber_sample_count = 0
            leaf_cropped_image_paths = get_image_paths_from_folder(leaf_cropped_dir)

            for cucumber_mask_path in cucumber_mask_paths:
                # 샘플링된 잎 이미지 가져오기
                sampled_leaf_paths = random_sample_leaf_paths(leaf_cropped_image_paths, samples_for_this_cucumber)

                for idx, leaf_cropped_image_path in enumerate(sampled_leaf_paths):
                    if sample_count >= sample_limit:
                        break  # 전체 샘플 한도 초과 시 종료
                    if cucumber_sample_count >= samples_for_this_cucumber:
                        break  # 오이 이미지의 샘플 한도 초과 시 종료

                    # 잎 이미지 쌍 생성
                    pair_idx = -(idx + 1)
                    leaves_cropped_image_paths = [leaf_cropped_image_path, sampled_leaf_paths[pair_idx]]

                    if random_ratio:
                        occlusion_ratio = occlusion_ratio_list[sample_count] / 100.0

                    # 오이와 잎 이미지를 합성
                    coco_json, global_image_id, global_annotation_id = process_amodal_images_and_masks(
                        cucumber_image_path=cucumber_image_path,
                        leaf_cropped_image_paths=leaves_cropped_image_paths,
                        cucumber_mask_path=cucumber_mask_path,
                        save_dir=save_dir,
                        mask_save_dir=mask_save_dir,
                        coco_json=coco_json,
                        global_image_id=global_image_id,
                        global_annotation_id=global_annotation_id,
                        position=position,
                        occlusion_ratio=occlusion_ratio,
                        initial_leaf_ratio=initial_leaf_ratio,
                        multi_leaves=multi_leaves
                    )

                    sample_count += 1
                    cucumber_sample_count += 1
                    pbar.update(1)

                    if sample_count >= sample_limit:
                        break

                if sample_count >= sample_limit:
                    break

            if sample_count >= sample_limit:
                break

    # 최종 COCO JSON 저장
    output_json_path = os.path.join(json_dir, "dataset.json")
    save_coco_json(coco_json, output_json_path)



    