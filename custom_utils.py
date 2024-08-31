# 이미지와 바운딩 박스를 한 번에 로드하는 함수
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import os
from typing import List, Tuple

def load_images_and_boxes(image_folder: str, label_folder: str, max_images: int = 10) -> Tuple[List[np.ndarray], List[List], List[str]]:
    images = []
    all_boxes = []
    image_names = []
    processed_count = 0

    # 레이블 파일 목록을 정렬된 상태로 가져옴
    label_files = sorted(os.listdir(label_folder))

    for label_file in label_files:
        if processed_count >= max_images:
            break

        # JSON 레이블 파일 읽기
        with open(os.path.join(label_folder, label_file), 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # 이미지 파일명 가져오기
        image_name = annotation['description']['image']
        image_path = os.path.join(image_folder, image_name)

        # 이미지 읽기
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 바운딩 박스 정보 가져오기
        boxes = []
        for result in annotation['result']:
            if result['type'] == 'bbox':
                x, y, w, h = result['x'], result['y'], result['w'], result['h']
                boxes.append([x, y, x + w, y + h])

        if not boxes:
            continue

        images.append(image_rgb)
        all_boxes.append(boxes)
        image_names.append(image_name)

        processed_count += 1

    return images, all_boxes, image_names

# 마스크와 이미지를 시각화하는 함수
def plot_masks_on_images(images: List[np.ndarray], masks: List[np.ndarray], image_names: List[str]):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    
    for idx, (image, mask_set, image_name) in enumerate(zip(images, masks, image_names)):
        for i in range(mask_set.shape[0]):  # 각 이미지당 여러 개의 마스크가 있을 수 있음
            ax = axes[idx // 5, idx % 5]
            ax.imshow(image)
            ax.imshow(mask_set[i], alpha=0.5, cmap='jet')
            ax.set_title(f"Image: {image_name} - Mask {i + 1}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# 마스크를 이미지 파일로 저장하는 함수
def save_masks(masks: List[np.ndarray], image_names: List[str], output_folder: str, target_size: Tuple[int, int] = (384, 512)):
    os.makedirs(output_folder, exist_ok=True)

    for mask_set, image_name in zip(masks, image_names):
        for i, mask in enumerate(mask_set):
            # 마스크 리사이즈
            resized_mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # 마스크 저장 경로 설정
            mask_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_mask_{i + 1}.png")
            
            # 마스크 저장
            cv2.imwrite(mask_path, (resized_mask * 255).astype(np.uint8))  # 마스크를 이진 이미지로 저장

            
def resize_images_and_masks(images: List[np.ndarray], masks: List[np.ndarray], target_size: Tuple[int, int] = (384, 512)) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    resized_images = []
    resized_masks = []

    for image, mask_set in zip(images, masks):
        # 이미지 리사이즈
        resized_image = cv2.resize(image, (target_size[1], target_size[0]))  # (width, height)로 설정
        resized_images.append(resized_image)

        # 마스크 리사이즈
        resized_mask_set = []
        for mask in mask_set:
            # 마스크를 uint8로 변환하여 리사이즈
            mask_uint8 = mask.astype(np.uint8)  # bool 타입을 uint8로 변환
            resized_mask = cv2.resize(mask_uint8, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            resized_mask_set.append(resized_mask)

        resized_masks.append(np.array(resized_mask_set))

    return resized_images, resized_masks