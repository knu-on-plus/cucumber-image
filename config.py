# config.py


# 하이퍼파라미터
HYPERPARAMETERS = {
    "saple_limit": 10000,
    "image_index_start": 0,  # 이미지 인덱스 시작값
    "position": "middle",  # 잎 위치 결정 옵션
    "occlusion_ratio": 0.5, 
}
# 경로 설정
INPUT_PATHS = {
    "cucumber_images_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/images/train",  # 오이 이미지 디렉토리
    "cucumber_masks_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/masks/train",  # 오이 마스크 디렉토리
    "leaf_cropped_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/cropped_leaves/train"  # 잎 객체 디렉토리
}

OUTPUT_PATHS = {
    "save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_images_resizing_test",  # 합성 이미지 저장 디렉토리
    "mask_save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/modal_masks_resizing_test",  # Modal 마스크 저장 디렉토리
    "json_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_info_resizing_test"  # Amodal 정보 JSON 저장 디렉토리
}


