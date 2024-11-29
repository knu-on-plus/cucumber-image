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
    마스크에서 일반적인 [x_min, y_min, x_max, y_max] 형식의 bbox 추출
    """
    coords = np.column_stack(np.where(mask == 255))  # 마스크 좌표 추출
    if coords.size == 0:  # 빈 마스크 처리
        return [0, 0, 0, 0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return [x_min, y_min, x_max, y_max]

def get_coco_bbox_from_mask(mask):
    """
    마스크에서 COCO 포맷 [x_min, y_min, width, height] 형식의 bbox 추출
    """
    x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

def mask_to_polygon(binary_mask, min_area=60):
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for contour in contours:
        # Simplify the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True).flatten().tolist()

        # Check if the contour is valid (minimum points and area)
        if len(contour) > 4 and cv2.contourArea(np.array(contour).reshape(-1, 2)) > min_area:
            polygons.append(contour)
    
    return polygons

def mask_to_polygon(binary_mask, min_contour_area=10):
    """
    Convert a binary mask to a COCO segmentation polygon, filtering by area.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:  # 최소 크기 필터링
            contour = contour.flatten().tolist()
            if len(contour) > 4:  # 최소한의 폴리곤 점이 필요
                polygons.append(contour)
    return polygons


def resize_image_and_masks(image, masks, target_size=(768, 1024)):
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]
    
    # 이미지 리사이즈
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # 마스크 리사이즈
    resized_masks = [
        cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]
    
    # 스케일 계산
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    return resized_image, resized_masks, scale_x, scale_y



def resize_leaf_image(leaf_image, cucumber_image):
    cucumber_h, cucumber_w = cucumber_image.shape[:2]
    leaf_h, leaf_w = leaf_image.shape[:2]

    # 잎 이미지가 오이 이미지를 초과하면 비율에 맞춰 리사이즈
    if leaf_h > cucumber_h or leaf_w > cucumber_w:
        scale = min(cucumber_h / leaf_h, cucumber_w / leaf_w)  # 가장 작은 스케일로 맞춤
        new_leaf_h = int(leaf_h * scale)
        new_leaf_w = int(leaf_w * scale)
        leaf_image = cv2.resize(leaf_image, (new_leaf_w, new_leaf_h), interpolation=cv2.INTER_AREA)
        print("leaves resized...")
    return leaf_image


import random

def calculate_leaf_location(cucumber_bbox, location='middle'):
    x_min, y_min, x_max, y_max = cucumber_bbox

    # 오이 중심 좌표 계산
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # 각 위치별 좌표 계산
    top = (center_x, y_min)        # 상단 중심
    middle = (center_x, center_y) # 중앙
    bottom = (center_x, y_max)    # 하단 중심

    # 위치 결정에 따른 결과 반환
    if location == 'top':
        return top
    elif location == 'middle':
        return middle
    elif location == 'bottom':
        return bottom
    elif location == 'random':
        return random.choice([top, bottom])  # 상단 또는 하단 중 랜덤
    else:
        raise ValueError(f"Invalid location option: {location}")


def resize_leaf_to_target_ratio(cucumber_mask, leaf_image, leaf_position, target_ratio):
    # 잎의 중심 좌표
    leaf_x, leaf_y = leaf_position
    cucumber_area = np.sum(cucumber_mask == 255)  # 오이의 총 면적 계산

    # 최소 및 최대 비율 제한
    min_ratio, max_ratio = 0.01, 0.9
    target_ratio = np.clip(target_ratio, min_ratio, max_ratio)
    
    # 잎의 초기 크기
    leaf_h, leaf_w = leaf_image.shape[:2]

    start_y = max(0, leaf_y - leaf_h // 2)
    start_x = max(0, leaf_x - leaf_w // 2)
    while True:
        # 잎 마스크 생성 (알파 채널 활용)
        leaf_mask = (leaf_image[:, :, 3] > 0).astype(np.uint8) * 255

        # 잎 마스크를 해당 위치로 이동
        temp_leaf_mask = np.zeros_like(cucumber_mask)
        end_y = min(cucumber_mask.shape[0], start_y + leaf_h)
        end_x = min(cucumber_mask.shape[1], start_x + leaf_w)

        # 잎 마스크 복사
        temp_leaf_mask[start_y:end_y, start_x:end_x] = leaf_mask[
            0:end_y - start_y, 0:end_x - start_x
        ]

        # 현재 겹침 영역 계산
        overlap_area = np.sum((temp_leaf_mask > 0) & (cucumber_mask > 0))
        current_ratio = overlap_area / cucumber_area

        # 목표 비율에 도달하면 종료
        if abs(current_ratio - target_ratio) < 0.01:
            break

        # 크기 조정 비율 계산
        scale_factor = (target_ratio / current_ratio) ** 0.5
        new_h = max(1, int(leaf_h * scale_factor))
        new_w = max(1, int(leaf_w * scale_factor))

        # 잎 이미지 크기 조정
        leaf_image = cv2.resize(leaf_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        leaf_h, leaf_w = leaf_image.shape[:2]  # 크기 업데이트

    return leaf_image


def merge_and_crop_leaf(cucumber_image, resized_leaf_image, leaf_position):
    # 잎의 중심 좌표와 크기
    leaf_x, leaf_y = leaf_position
    leaf_h, leaf_w = resized_leaf_image.shape[:2]

    # 잎 이미지의 좌측 상단 좌표 계산
    crop_x_start = max(0, leaf_x - leaf_w // 2)
    crop_y_start = max(0, leaf_y - leaf_h // 2)

    # 잎 이미지를 경계에 맞게 자르기
    leaf_crop_start_y = max(0, -leaf_y + leaf_h // 2)  # 잎의 시작 Y 좌표
    leaf_crop_end_y = leaf_h - max(0, (leaf_y + leaf_h // 2) - cucumber_image.shape[0])
    leaf_crop_start_x = max(0, -leaf_x + leaf_w // 2)  # 잎의 시작 X 좌표
    leaf_crop_end_x = leaf_w - max(0, (leaf_x + leaf_w // 2) - cucumber_image.shape[1])

    cropped_leaf_image = resized_leaf_image[
        leaf_crop_start_y:leaf_crop_end_y,
        leaf_crop_start_x:leaf_crop_end_x
    ]

    # 결과 이미지와 마스크 초기화
    merged_image = cucumber_image.copy()
    leaf_mask = np.zeros((cucumber_image.shape[0], cucumber_image.shape[1]), dtype=np.uint8)

    # 잎 이미지 병합
    cropped_h, cropped_w = cropped_leaf_image.shape[:2]
    for i in range(cropped_h):
        for j in range(cropped_w):
            if cropped_leaf_image[i, j, 3] > 0:  # 투명하지 않은 경우
                merged_image[crop_y_start + i, crop_x_start + j] = cropped_leaf_image[i, j, :3]
                leaf_mask[crop_y_start + i, crop_x_start + j] = 255  # 잎 마스크 업데이트

    return merged_image, leaf_mask




