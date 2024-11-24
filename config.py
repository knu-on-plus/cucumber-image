# config.py


# 하이퍼파라미터
HYPERPARAMETERS = {
    "saple_limit": 40,
    "offset_y_range": (-50, 50),  # offset_y의 무작위 범위
    "image_index_start": 0  # 이미지 인덱스 시작값
}
# 경로 설정
INPUT_PATHS = {
    "cucumber_images_dir": "/home/knuvi/Desktop/song/cucumber-image/data/oi_seg/images",  # 오이 이미지 디렉토리
    "cucumber_masks_dir": "/home/knuvi/Desktop/song/cucumber-image/data/oi_seg/masks",  # 오이 마스크 디렉토리
    "leaf_cropped_dir": "/home/knuvi/Desktop/song/cucumber-image/data/oi_seg/selected_cropped_leaves2"  # 잎 객체 디렉토리
}

OUTPUT_PATHS = {
    "save_dir": "/home/knuvi/Desktop/song/cucumber-image/data/oi_seg/amodal_images7",  # 합성 이미지 저장 디렉토리
    "mask_save_dir": "/home/knuvi/Desktop/song/cucumber-image/data/oi_seg/modal_masks7",  # Modal 마스크 저장 디렉토리
    "json_dir": "/home/knuvi/Desktop/song/cucumber-image/data/oi_seg/amodal_info7"  # Amodal 정보 JSON 저장 디렉토리
}


