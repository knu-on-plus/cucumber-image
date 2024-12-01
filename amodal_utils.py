import json
import os, sys
import cv2
import numpy as np
from custom_utils import *
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

def leaf_size_initialization(cucumber_mask, ratio = (0.4, 0.25)):
    max_leaf_size_ratio, min_leaf_size_ratio = ratio  # 잎의 최대 및 최소 크기 비율
    max_leaf_h, max_leaf_w = int(cucumber_mask.shape[0] * max_leaf_size_ratio), int(cucumber_mask.shape[1] * max_leaf_size_ratio)
    min_leaf_h, min_leaf_w = int(cucumber_mask.shape[0] * min_leaf_size_ratio), int(cucumber_mask.shape[1] * min_leaf_size_ratio)

    return max_leaf_h, max_leaf_w, min_leaf_h, min_leaf_w

def resize_leaf_to_target_ratio(cucumber_mask, leaf_image, leaf_position, target_ratio):
    loss_rate = 0.05  # 손실률
    # 잎의 중심 좌표
    leaf_x, leaf_y = leaf_position
    cucumber_area = np.sum(cucumber_mask == 255)  # 오이 마스크 면적 계산

    # 초기 잎 크기 제한 (오이 이미지 크기 비율에 따라)
    max_leaf_size_ratio, min_leaf_size_ratio = 0.4, 0.25  # 잎의 최대 크기를 오이 이미지의 50%로 제한
    max_leaf_h, max_leaf_w, min_leaf_h, min_leaf_w = leaf_size_initialization(cucumber_mask, ratio=(max_leaf_size_ratio, min_leaf_size_ratio) )  # 잎 크기 초기화

    # 잎 이미지 크기 제한
    leaf_h, leaf_w = leaf_image.shape[:2]
    if leaf_h > max_leaf_h or leaf_w > max_leaf_w:
        leaf_image = cv2.resize(leaf_image, (max_leaf_w, max_leaf_h), interpolation=cv2.INTER_LINEAR)
    elif leaf_h < min_leaf_h or leaf_w < min_leaf_w:
        leaf_image = cv2.resize(leaf_image, (min_leaf_w, min_leaf_h), interpolation=cv2.INTER_LINEAR)

    leaf_h, leaf_w = leaf_image.shape[:2]
    start_y = max(0, leaf_y - leaf_h // 2)
    start_x = max(0, leaf_x - leaf_w // 2)

    max_iterations = 200
    iterations = 0

    while True:
        # 잎 마스크 생성 (알파 채널 활용)
        leaf_mask = (leaf_image[:, :, 3] > 0).astype(np.uint8) * 255

        # 잎 마스크를 해당 위치로 이동
        temp_leaf_mask = np.zeros_like(cucumber_mask)
        end_y = min(cucumber_mask.shape[0], start_y + leaf_h)
        end_x = min(cucumber_mask.shape[1], start_x + leaf_w)

        temp_leaf_mask[start_y:end_y, start_x:end_x] = leaf_mask[0:end_y - start_y, 0:end_x - start_x]

        # 현재 겹침 영역 계산
        overlap_area = np.sum((temp_leaf_mask > 0) & (cucumber_mask > 0))
        current_ratio = overlap_area / cucumber_area
        
        # 디버깅 출력
        print(f"Debug: Cucumber Area: {cucumber_area}, Overlap Area: {overlap_area}, Leaf Area: {leaf_h * leaf_w}, Current Ratio: {current_ratio:.4f}")

        # 목표 비율에 도달하면 종료
        if abs(current_ratio - target_ratio) < loss_rate and iterations > 0:
            print(f"Target ratio achieved with current ratio: {current_ratio:.4f} after {iterations} iterations.")
            break

        # 반복 초과 시 종료
        if iterations >= max_iterations:
            print(f"Error: Maximum iterations ({max_iterations}) reached. Current ratio: {current_ratio:.4f}. Exiting.")
            exit(1)

        # 크기 조정 비율 계산
        if current_ratio < target_ratio:
            scale_factor = min((target_ratio / current_ratio) ** 0.5, 1.02)
        else:
            scale_factor = max((target_ratio / current_ratio) ** 0.5, 0.98)
        
        
        new_h, new_w = max(1, int(leaf_h * scale_factor)), max(1, int(leaf_w * scale_factor))
        
        # 잎 이미지 크기 조정
        leaf_image = cv2.resize(leaf_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        leaf_h, leaf_w = leaf_image.shape[:2]  # 크기 업데이트
        iterations += 1

    print("Leaves resized to target ratio.")
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
            # 경계 초과 방지 조건
            if (crop_y_start + i >= cucumber_image.shape[0]) or (crop_x_start + j >= cucumber_image.shape[1]):
                print("====== Warning: Leaf image out of bounds. ======")
                continue  # 경계를 벗어난 경우 스킵

            if cropped_leaf_image[i, j, 3] > 0:  # 투명하지 않은 경우
                merged_image[crop_y_start + i, crop_x_start + j] = cropped_leaf_image[i, j, :3]
                leaf_mask[crop_y_start + i, crop_x_start + j] = 255  # 잎 마스크 업데이트

    return merged_image, leaf_mask


# modal 마스크 생성
def generate_save_masks(cucumber_mask, leaf_mask, image_name, save_dir):
    # 겹치는 부분 (오이 마스크와 잎 마스크가 동시에 255인 부분을 추출)
    overlap_mask = (cucumber_mask == 255) & (leaf_mask == 255)
    
    # Modal 마스크 생성 (겹치는 부분을 제외한 오이 마스크)
    modal_mask = cucumber_mask.copy()
    modal_mask[overlap_mask] = 0  # overlap 영역을 0으로 만들어 겹친 부분 제거

    # 마스크를 PNG 형식으로 저장 (이진화된 값 0 또는 255로 저장)
    modal_filename = f"{os.path.splitext(image_name)[0]}_amodal_mask.png"
    amodal_mask_path = save_image(save_dir, modal_filename, cucumber_mask)

    # 마스크를 PNG 형식으로 저장 (이진화된 값 0 또는 255로 저장)
    modal_filename = f"{os.path.splitext(image_name)[0]}_modal_mask.png"
    modal_mask_path = save_image(save_dir, modal_filename, modal_mask)

    # Overlap 마스크 저장
    overlap_filename = f"{os.path.splitext(image_name)[0]}_overlap_mask.png"
    overlap_mask_path = save_image(save_dir, overlap_filename, overlap_mask.astype(np.uint8) * 255)

    return modal_mask, overlap_mask

