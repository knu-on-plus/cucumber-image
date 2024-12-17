import json
import os, sys
import cv2
import numpy as np
from custom_utils import *
import random
import matplotlib.pyplot as plt


def calculate_leaf_location(cucumber_bbox, location='middle'):
    x_min, y_min, x_max, y_max = cucumber_bbox

    # 오이 중심 좌표 계산
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # 각 위치별 좌표 계산
    top = (center_x, int(y_min + (y_max - y_min) * 0.10))      # 상단 중심
    middle = (center_x, center_y) # 중앙
    bottom = (center_x, int(y_min + (y_max - y_min) * 0.80))   # 하단 중심

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

def leaf_size_initialization(cucumber_mask, leaf_image, ratio = (0.35, 0.5)):
    min_leaf_size_ratio, max_leaf_size_ratio = ratio  # 잎의 최대 및 최소 크기 비율
    max_leaf_h, max_leaf_w = int(cucumber_mask.shape[0] * max_leaf_size_ratio), int(cucumber_mask.shape[1] * max_leaf_size_ratio)
    min_leaf_h, min_leaf_w = int(cucumber_mask.shape[0] * min_leaf_size_ratio), int(cucumber_mask.shape[1] * min_leaf_size_ratio)

    # 잎 이미지 크기 제한
    leaf_h, leaf_w = leaf_image.shape[:2]
    if leaf_h > max_leaf_h or leaf_w > max_leaf_w:
        leaf_image = cv2.resize(leaf_image, (max_leaf_w, max_leaf_h), interpolation=cv2.INTER_LINEAR)
    elif leaf_h < min_leaf_h or leaf_w < min_leaf_w:
        leaf_image = cv2.resize(leaf_image, (min_leaf_w, min_leaf_h), interpolation=cv2.INTER_LINEAR)

    return leaf_image

def resize_leaf_to_target_ratio(cucumber_mask, leaf_image, leaf_position, target_ratio):
    loss_rate = 0.05  # 허용 오차
    leaf_x, leaf_y = leaf_position
    cucumber_area = np.sum(cucumber_mask == 255)  # 오이 마스크 면적 계산

    leaf_h, leaf_w = leaf_image.shape[:2]
    max_iterations = 10
    iterations = 0

    while True:
        # 잎 마스크 생성 (알파 채널 활용)
        leaf_mask = (leaf_image[:, :, 3] > 0).astype(np.uint8) * 255

        # 중심을 leaf_position으로 이동
        temp_leaf_mask = np.zeros_like(cucumber_mask)
        start_y = max(0, leaf_y - leaf_h // 2)
        start_x = max(0, leaf_x - leaf_w // 2)
        end_y = min(cucumber_mask.shape[0], start_y + leaf_h)
        end_x = min(cucumber_mask.shape[1], start_x + leaf_w)

        temp_leaf_mask[start_y:end_y, start_x:end_x] = leaf_mask[0:(end_y - start_y), 0:(end_x - start_x)]

        # 현재 겹침 영역 계산
        overlap_area = np.sum((temp_leaf_mask > 0) & (cucumber_mask > 0))
        current_ratio = overlap_area / cucumber_area

        # 시각화
        #visualize_resizing(cucumber_mask, temp_leaf_mask, leaf_position, overlap_area, current_ratio, iterations)

        # 디버깅 출력
        #print(f"Iteration {iterations}: Overlap Area: {overlap_area}, Leaf Area: {leaf_h * leaf_w}, Current Ratio: {current_ratio:.4f}")

        # 목표 비율에 도달하면 종료
        if abs(current_ratio - target_ratio) < loss_rate:
            #print(f"Target ratio achieved with current ratio: {current_ratio:.4f} after {iterations} iterations.")
            break

        # 반복 초과 시 종료
        if iterations >= max_iterations:
            #print(f"Error: Maximum iterations ({max_iterations}) reached. Exiting.")
            break

        # 크기 조정 비율 계산
        if current_ratio < target_ratio:
            scale_factor = min((target_ratio / current_ratio) ** 0.5, 1.1)
        else:
            scale_factor = max((target_ratio / current_ratio) ** 0.5, 0.9)

        # Resize된 크기 계산
        new_h = max(1, int(leaf_h * scale_factor))
        new_w = max(1, int(leaf_w * scale_factor))

        # Resize 수행
        resized_leaf = cv2.resize(leaf_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 업데이트
        leaf_image = resized_leaf
        leaf_h, leaf_w = leaf_image.shape[:2]
        iterations += 1

    #print("Leaves resized to target ratio.")
    return leaf_image

def adjust_leaves_to_occlusion(cucumber_mask, leaf_image1, leaf_image2, leaf_location1, leaf_location2, target_ratio):
    def create_leaf_mask(leaf_image, leaf_position, mask_shape):
        leaf_h, leaf_w = leaf_image.shape[:2]
        temp_mask = np.zeros(mask_shape, dtype=np.uint8)

        # 알파 채널로 잎 마스크 생성
        leaf_mask = (leaf_image[:, :, 3] > 0).astype(np.uint8) * 255

        # 잎 중심점 기준으로 시작/끝 좌표 계산
        leaf_x, leaf_y = leaf_position
        start_y = leaf_y - leaf_h // 2
        start_x = leaf_x - leaf_w // 2
        end_y = leaf_y + leaf_h // 2
        end_x = leaf_x + leaf_w // 2

        # 경계 조건 처리: 오이 이미지를 벗어나는 경우 잘라내기
        crop_start_y = max(0, -start_y)
        crop_start_x = max(0, -start_x)
        crop_end_y = leaf_h - max(0, end_y - mask_shape[0])
        crop_end_x = leaf_w - max(0, end_x - mask_shape[1])

        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(mask_shape[0], end_y)
        end_x = min(mask_shape[1], end_x)

        # 크기 일치 확인 후 복사
        region_h = end_y - start_y
        region_w = end_x - start_x
        crop_h = crop_end_y - crop_start_y
        crop_w = crop_end_x - crop_start_x

        if region_h != crop_h or region_w != crop_w:
            # 크기 일치하도록 잘라냄
            region_h = min(region_h, crop_h)
            region_w = min(region_w, crop_w)

            end_y = start_y + region_h
            end_x = start_x + region_w
            crop_end_y = crop_start_y + region_h
            crop_end_x = crop_start_x + region_w

        # 마스크 복사 및 잘라내기
        temp_mask[start_y:end_y, start_x:end_x] = leaf_mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        # 중심점 디버깅
        #print(f"Leaf Position: {leaf_position}, Mask Center Adjusted to: ({leaf_x}, {leaf_y})")
        return temp_mask

    def move_leaf(leaf_position, direction, step, mask_shape):
        x, y = leaf_position
        height, width = mask_shape

        if direction == "up":
            new_y = max(0, y - step)  # 상단 경계를 벗어나지 않도록 제한
        elif direction == "down":
            new_y = min(height - 1, y + step)  # 하단 경계를 벗어나지 않도록 제한
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'up' or 'down'.")

        return (x, new_y)

    max_iterations = 20
    loss_rate = 0.05
    step_size = 10
    iterations = 0

    cucumber_area = np.sum(cucumber_mask > 0)

    while iterations < max_iterations:
        # 잎 마스크 생성
        leaf_mask1 = create_leaf_mask(leaf_image1, leaf_location1, cucumber_mask.shape)
        leaf_mask2 = create_leaf_mask(leaf_image2, leaf_location2, cucumber_mask.shape)

        # 두 잎 마스크 병합
        combined_leaf_mask = cv2.bitwise_or(leaf_mask1, leaf_mask2)

        # 겹침 비율 계산
        overlap_area = np.sum((cucumber_mask > 0) & (combined_leaf_mask > 0))
        current_ratio = overlap_area / cucumber_area

        #print(f"Iteration {iterations}: , Current Ratio: {current_ratio:.4f}")
        #visualize_shifting(cucumber_mask, leaf_mask1, leaf_mask2, leaf_location1, leaf_location2, iterations)

        # 목표 비율 도달 여부 확인
        if abs(current_ratio - target_ratio) <= loss_rate:
            #print(f"Target ratio met: {current_ratio:.4f}")
            break

        # 비율에 따라 중심점 이동
        if current_ratio < target_ratio:
            leaf_location1 = move_leaf(leaf_location1, "down", step_size, cucumber_mask.shape)
            leaf_location2 = move_leaf(leaf_location2, "up", step_size, cucumber_mask.shape)
        else:
            leaf_location1 = move_leaf(leaf_location1, "up", step_size, cucumber_mask.shape)
            leaf_location2 = move_leaf(leaf_location2, "down", step_size, cucumber_mask.shape)

        iterations += 1

    #if iterations >= max_iterations:
        #print(f"Warning: Maximum iterations reached. Final ratio: {current_ratio:.4f}")

    return leaf_location1, leaf_location2

def overlap_dual_leaves(cucumber_mask, leaf_image1, leaf_image2, initial_leaf_ratio):
    leaf_image1 = leaf_size_initialization(cucumber_mask, leaf_image1, initial_leaf_ratio)
    leaf_image2 = leaf_size_initialization(cucumber_mask, leaf_image2, initial_leaf_ratio)
    # 이미지 크기 비교
    h1, w1, _ = leaf_image1.shape
    h2, w2, _ = leaf_image2.shape

    if w1 < w2:  # leaf_image2가 더 크다면 위치를 바꿈
        leaf_image1, leaf_image2 = leaf_image2, leaf_image1
        h1, w1, h2, w2 = h2, w2, h1, w1

    # 새 캔버스 크기 계산
    canvas_width = w1 + w2 // 2
    canvas_height = max(h1, h2)

    # 새로운 캔버스 생성 (RGBA)
    overlapped_leaves = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # leaf_image1 배치 (왼쪽)
    x1_start, y1_start = 0, (canvas_height - h1) // 2
    overlapped_leaves[y1_start:y1_start + h1, x1_start:x1_start + w1, :] = leaf_image1

    # leaf_image2 배치 (오른쪽)
    x2_start = max(0, min(w1 - w2 // 2, canvas_width - w2))  # 오른쪽 끝 초과 방지
    y2_start = max(0, (canvas_height - h2) // 2)

    for c in range(4):  # 채널별 병합 (RGBA)
        overlapped_leaves[y2_start:y2_start + h2, x2_start:x2_start + w2, c] = np.where(
            leaf_image2[:, :, 3] > 0,  # 알파 채널이 있는 경우
            leaf_image2[:, :, c],
            overlapped_leaves[y2_start:y2_start + h2, x2_start:x2_start + w2, c]
        )
    return overlapped_leaves

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
                #print("====== Warning: Leaf image out of bounds. ======")
                continue  # 경계를 벗어난 경우 스킵

            if cropped_leaf_image[i, j, 3] > 0:  # 투명하지 않은 경우
                merged_image[crop_y_start + i, crop_x_start + j] = cropped_leaf_image[i, j, :3]
                leaf_mask[crop_y_start + i, crop_x_start + j] = 255  # 잎 마스크 업데이트

    return merged_image, leaf_mask

def save_processed_masks(amodal_mask, overlap_mask, modal_mask, occluder_mask, image_name, mask_save_dir):
    
    # 마스크를 PNG 형식으로 저장 (이진화된 값 0 또는 255로 저장)
    modal_filename = f"{os.path.splitext(image_name)[0]}_occluder_mask.png"
    occluder_mask_path = save_image(mask_save_dir, modal_filename, amodal_mask)
    
    # 마스크를 PNG 형식으로 저장 (이진화된 값 0 또는 255로 저장)
    modal_filename = f"{os.path.splitext(image_name)[0]}_amodal_mask.png"
    amodal_mask_path = save_image(mask_save_dir, modal_filename, amodal_mask)

    # 마스크를 PNG 형식으로 저장 (이진화된 값 0 또는 255로 저장)
    modal_filename = f"{os.path.splitext(image_name)[0]}_modal_mask.png"
    modal_mask_path = save_image(mask_save_dir, modal_filename, modal_mask)

    # Overlap 마스크 저장
    overlap_filename = f"{os.path.splitext(image_name)[0]}_overlap_mask.png"
    overlap_mask_path = save_image(mask_save_dir, overlap_filename, overlap_mask.astype(np.uint8) * 255)


# modal 마스크 생성
def get_amodal_masks(cucumber_mask, leaf_mask):
    # 겹치는 부분 (오이 마스크와 잎 마스크가 동시에 255인 부분을 추출)
    overlap_mask = (cucumber_mask == 255) & (leaf_mask == 255)
    
    # Modal 마스크 생성 (겹치는 부분을 제외한 오이 마스크)
    modal_mask = cucumber_mask.copy()
    modal_mask[overlap_mask] = 0  # overlap 영역을 0으로 만들어 겹친 부분 제거

    return modal_mask, overlap_mask


def generate_annotation(amodal_mask, modal_mask, global_id, image_id, category_id, occluder_segm=[]):
    amodal_segm = mask_to_polygon(amodal_mask)
    amodal_bbox = get_coco_bbox_from_mask(amodal_mask)
    
    process_mask = modal_mask if modal_mask is not None else amodal_mask  # 보이는 영역이 없으면 전체 영역(amodal_mask) 사용
    area = float(np.sum(process_mask == 255))
    segmentation = visible_segm = mask_to_polygon(process_mask)
    visible_bbox = get_coco_bbox_from_mask(process_mask)

    annotation = {
        "id": global_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "amodal_bbox": amodal_bbox,
        "visible_bbox" : visible_bbox,
        "bbox": amodal_bbox,  # COCO 포맷 BBox 계산
        "area": area,
        "amodal_area": float(np.sum(amodal_mask == 255)),
        "amodal_segm": amodal_segm,
        "segmentation": amodal_segm,  # Segmentation polygon 생성
        "visible_segm": visible_segm,
        "background_objs_segm": [],  # 기본값
        "occluder_segm": occluder_segm,
    }
    return annotation

def create_occlusion_ratio_list(sample_limit, ratios, proportions):
    """
    ratios: 리스트, e.g., [50, 75, 90]
    proportions: 리스트, e.g., [5, 4, 1]
    """
    total_proportion = sum(proportions)
    counts = [int(sample_limit * p / total_proportion) for p in proportions]
    
    # 마지막 비율에 남은 샘플 모두 할당
    counts[-1] = sample_limit - sum(counts[:-1])
    
    occlusion_ratios = []
    for ratio, count in zip(ratios, counts):
        occlusion_ratios.extend([ratio] * count)
    
    random.shuffle(occlusion_ratios)
    return occlusion_ratios

def visualize_resizing(cucumber_mask, temp_leaf_mask, leaf_position, overlap_area, current_ratio, iteration):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # 원본 cucumber_mask 시각화
    ax[0].imshow(cucumber_mask, cmap="gray")
    ax[0].scatter(leaf_position[0], leaf_position[1], color='red', label='Leaf Position')
    ax[0].set_title("Original Cucumber Mask")
    ax[0].legend()

    # Overlap된 영역 시각화
    overlap_visual = cucumber_mask.copy()
    overlap_visual[temp_leaf_mask > 0] = 255  # Overlap 영역 강조
    ax[1].imshow(overlap_visual, cmap="Reds")
    ax[1].scatter(leaf_position[0], leaf_position[1], color='red', label='Leaf Position')

    # 계산된 중심을 시각화
    ax[1].set_title(f"Iteration {iteration}\nOverlap Area: {overlap_area}, Ratio: {current_ratio:.4f}")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def visualize_shifting(cucumber_mask, leaf_mask1, leaf_mask2, leaf_location1, leaf_location2, iteration):
    # 겹침 영역 계산
    overlap_area = cv2.bitwise_and(cucumber_mask, cv2.bitwise_or(leaf_mask1, leaf_mask2))
    
    # 시각화 준비
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 전체 배경 생성
    visualization = np.zeros_like(cucumber_mask, dtype=np.uint8)
    visualization[cucumber_mask > 0] = 128  # 오이 객체는 회색
    visualization[leaf_mask1 > 0] = 200  # 잎 1은 밝은 회색
    visualization[leaf_mask2 > 0] = 255  # 잎 2는 흰색
    visualization[overlap_area > 0] = 50  # 겹치는 영역은 어두운 색으로 표시
    
    # 시각화
    ax.imshow(visualization, cmap="gray")
    ax.scatter(leaf_location1[0], leaf_location1[1], color='red', label='Leaf 1 Center')
    ax.scatter(leaf_location2[0], leaf_location2[1], color='blue', label='Leaf 2 Center')
    ax.set_title(f"Iteration {iteration}: Overlap Visualization")
    ax.legend()
    
    plt.show()

