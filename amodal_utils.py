import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def visualize_merged_amodal_and_modal(merged_image, amodal_mask, modal_mask):
    """
    합성된 이미지, Amodal 마스크, Modal 마스크를 시각화하는 함수
    :param merged_image: 합성된 이미지 (RGB)
    :param amodal_mask: Amodal 마스크 (오이 전체 마스크)
    :param modal_mask: Modal 마스크 (잎에 의해 가려진 오이 마스크)
    """
    
    # Plot 설정
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))  # 1행 3열 레이아웃
    
    # 합성 이미지
    axs[0].imshow(cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환
    axs[0].set_title('Merged Image')
    axs[0].axis('off')

    # Amodal 마스크
    axs[1].imshow(amodal_mask, cmap='gray')
    axs[1].set_title('Amodal Mask')
    axs[1].axis('off')

    # Modal 마스크
    axs[2].imshow(modal_mask, cmap='gray')
    axs[2].set_title('Modal Mask')
    axs[2].axis('off')

    plt.subplots_adjust(wspace=0.1)  # 가로 간격을 0.1로 줄이기
    plt.tight_layout()
    plt.show()

def visualize_all_masks(cucumber_mask, leaf_mask, overlap_mask):
    
    plt.figure(figsize=(10, 5))
    
    # 오이 마스크 시각화
    plt.subplot(1, 3, 1)
    plt.imshow(cucumber_mask, cmap='gray')
    plt.title("Cucumber Mask")
    
    # 잎 마스크 시각화
    plt.subplot(1, 3, 2)
    plt.imshow(leaf_mask, cmap='gray')
    plt.title("Leaf Mask")
    
    # 겹치는 부분(overlap) 시각화
    plt.subplot(1, 3, 3)
    plt.imshow(overlap_mask, cmap='gray')
    plt.title("Modal Mask")
    
    plt.tight_layout()
    plt.show()