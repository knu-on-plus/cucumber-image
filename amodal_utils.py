import json
import os
import cv2
import numpy as np

def get_image_paths_from_folder(folder_path, extensions=['.jpg', '.png']):
    image_paths = []
    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in extensions):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

def save_image(save_dir, file_name, img, flag=0):
    key = 'Mask' if flag == 0 else 'Image'

    os.makedirs(save_dir, exist_ok=True)  # 저장 경로가 없으면 생성
    image_save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(image_save_path, img)
    print(f"{key} 저장됨: {image_save_path}")
    return image_save_path

def get_bbox_from_mask(mask):
    """
    마스크에서 bbox 추출
    :param mask: 오이 마스크 (0과 255로 구성된 이진 마스크)
    :return: bbox [x_min, y_min, x_max, y_max]
    """
    # 마스크에서 255인 영역 (오이) 좌표 찾기
    coords = np.column_stack(np.where(mask == 255))
    
    # 좌상단 (y_min, x_min)과 우하단 (y_max, x_max) 좌표 계산
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # bbox 반환 [x_min, y_min, x_max, y_max]
    return [x_min, y_min, x_max, y_max]


def mask_to_polygon(binary_mask):
    """
    Convert a binary mask to a COCO segmentation polygon.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # 최소한의 폴리곤 점이 필요
            polygons.append(contour)
    return polygons


def convert_to_serializable(data):
    """
    JSON으로 직렬화 가능한 Python 기본 데이터 타입으로 변환.
    :param data: 변환할 데이터
    :return: Python 기본 데이터 타입
    """
    if isinstance(data, np.ndarray):
        return data.tolist()  # numpy 배열을 리스트로 변환
    elif isinstance(data, (np.int64, np.int32, np.int16)):
        return int(data)  # numpy 정수를 Python 정수로 변환
    elif isinstance(data, (np.float64, np.float32, np.float16)):
        return float(data)  # numpy 실수를 Python 실수로 변환
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)  # numpy 논리를 Python 논리로 변환
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}  # 재귀 처리
    elif isinstance(data, list):
        return [convert_to_serializable(v) for v in data]  # 리스트 항목 재귀 처리
    else:
        return data  # 이미 직렬화 가능한 경우 반환