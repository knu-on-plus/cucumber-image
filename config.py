# config.py


# 하이퍼파라미터
HYPERPARAMETERS = {
    "dataset_type": 'debugging', ## [train, valid, debugging]
    "image_index_start": 0,  # 이미지 인덱스 시작값
    "position": "middle",  # 잎 위치 결정 옵션 (top, middle, bottom, random)
    "occlusion_ratio": 0.5,
    "multi_leaves": 0,  # 다수의 잎 사용 여부 [0: False, 1: top, bottom dual leaves, 2: overlap dual leaves]
    'random_ratio': True,  # Random ratio 
    "initial_leaf_ratio": (0.20, 0.4),  # 잎 초기 크기 비율
    "r_settings": [50, 70, 90],  # Random ratio settings #(0.5, 0.75, 0.9)
    "r_proportions": [5, 4, 1],  # Random ratio proportions
    "sort": True,
}
# baseline position: middle, occlusion_ratio: 0.5, multi_leaves: 0, random_ratio: False, initial_leaf_ratio: (0.20, 0.4), r_settings: [50, 75, 90], r_proportions: [5, 4, 1]
# condition 1: position: random, occlusion_ratio: 0.5, multi_leaves: 0, random_ratio: True, initial_leaf_ratio: (0.20, 0.4), r_settings: [50, 75, 90], r_proportions: [5, 4, 1]
# condition 2: position: random, occlusion_ratio: 0.5, multi_leaves: 1, random_ratio: True, initial_leaf_ratio: (0.20, 0.4), r_settings: [50, 75, 90], r_proportions: [5, 4, 1]
# condition 3: position: random, occlusion_ratio: 0.5, multi_leaves: 2, random_ratio: True, initial_leaf_ratio: (0.20, 0.4), r_settings: [50, 75, 90], r_proportions: [5, 4, 1]

# 경로 설정
INPUT_PATHS = {
    "cucumber_images_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/images/",  # 오이 이미지 디렉토리
    "cucumber_masks_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/masks/",  # 오이 마스크 디렉토리
    "leaf_cropped_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/cropped_leaves/"  # 잎 객체 디렉토리
}

OUTPUT_PATHS = {
    "save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_images_baseline_test2",  # 합성 이미지 저장 디렉토리
    "mask_save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/modal_masks_baseline_test2",  # Modal 마스크 저장 디렉토리
    "json_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_info_baseline_test2"  # Amodal 정보 JSON 저장 디렉토리
}


