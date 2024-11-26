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
