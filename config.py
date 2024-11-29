# config.py


# 하이퍼파라미터
HYPERPARAMETERS = {
    "saple_limit": 4000,
    "image_index_start": 0,  # 이미지 인덱스 시작값
    "paste_location": "middle",  # 잎 위치 결정 옵션
}
# 경로 설정
INPUT_PATHS = {
    "cucumber_images_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/images/valid",  # 오이 이미지 디렉토리
    "cucumber_masks_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/masks/valid",  # 오이 마스크 디렉토리
    "leaf_cropped_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/splitted/cropped_leaves/valid"  # 잎 객체 디렉토리
}

OUTPUT_PATHS = {
    "save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_images_valid",  # 합성 이미지 저장 디렉토리
    "mask_save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/modal_masks_valid",  # Modal 마스크 저장 디렉토리
    "json_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_info_valid"  # Amodal 정보 JSON 저장 디렉토리
}


