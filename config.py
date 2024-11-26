# config.py


# 하이퍼파라미터
HYPERPARAMETERS = {
    "saple_limit": 20,
    "offset_y_range": (-50, 50),  # offset_y의 무작위 범위
    "image_index_start": 0  # 이미지 인덱스 시작값
}
# 경로 설정
INPUT_PATHS = {
    "cucumber_images_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/images",  # 오이 이미지 디렉토리
    "cucumber_masks_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/masks",  # 오이 마스크 디렉토리
    "leaf_cropped_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/cropped_objects/leaves"  # 잎 객체 디렉토리
}

OUTPUT_PATHS = {
    "save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_images6",  # 합성 이미지 저장 디렉토리
    "mask_save_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/modal_masks6",  # Modal 마스크 저장 디렉토리
    "json_dir": "/home/knuvi/Desktop/song/occlusion-mask-generation/data/synthesis/amodal_info6"  # Amodal 정보 JSON 저장 디렉토리
}


