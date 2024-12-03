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

def ensure_directories_exist(directories):
    """
    주어진 경로 목록에 해당하는 디렉터리가 존재하지 않을 경우 생성합니다.

    :param directories: 디렉터리 경로 목록 (리스트 또는 딕셔너리)
    """
    if isinstance(directories, dict):
        directories = directories.values()  # 딕셔너리 값들을 목록으로 변환
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"디렉터리 생성됨: {directory}")
        else:
            print(f"디렉터리가 이미 존재합니다: {directory}")


def save_image(save_dir, file_name, img, flag=0):
    key = 'Mask' if flag == 0 else 'Image'

    os.makedirs(save_dir, exist_ok=True)  # 저장 경로가 없으면 생성
    image_save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(image_save_path, img)
    print(f"{key} 저장됨: {image_save_path}")
    return image_save_path

def get_image_paths_from_folder(folder_path, extensions=['.jpg', '.png'], sort=False):
    image_paths = []
    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in extensions):
            image_paths.append(os.path.join(folder_path, filename))
    
    if sort:
        image_paths.sort()  # 정렬 수행
    
    return image_paths



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
    
    return resized_image, resized_masks



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