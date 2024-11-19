# coco_format.py
def create_empty_coco_format():
    """
    COCO JSON 기본 구조를 생성합니다. 데이터는 비어 있는 상태입니다.
    """
    return {
        "info": {
            "date_created": "",       # 생성 날짜
            "description": ""         # 데이터셋 설명
        },
        "images": [
            # 이미지 메타정보 리스트
            {
                "id": 0,              # 이미지 ID
                "width": 0,           # 이미지 너비
                "height": 0,          # 이미지 높이
                "file_name": ""       # 이미지 파일 이름
            }
        ],
        "annotations": [
            # 주석 정보 리스트
            {
                "id": 0,              # 주석 ID
                "image_id": 0,        # 연결된 이미지 ID
                "category_id": 0,     # 객체 클래스 ID
                "amodal_bbox": [],    # Amodal Bounding Box
                "visible_bbox": [],   # Visible Bounding Box
                "bbox": [],           # 기본 Bounding Box
                "amodal_area": 0.0,   # Amodal 영역 면적
                "area": 0.0,          # Visible 영역 면적
                "amodal_segm": [],    # Amodal Segmentation
                "segmentation": [],   # Visible Segmentation
                "visible_segm": [],   # 보이는 Segmentation
                "occluder_segm": [],  # 가리는 객체 Segmentation
                "background_objs_segm": []  # 배경 객체 Segmentation
            }
        ],
        "categories": [
            # 카테고리 리스트
            {
                "id": 0,             # 카테고리 ID
                "name": "",          # 카테고리 이름
                "supercategory": ""  # 상위 카테고리
            }
        ],
        "licenses": [
            # 라이선스 정보 리스트
            {
                "id": 0,             # 라이선스 ID
                "name": "",          # 라이선스 이름
                "url": ""            # 라이선스 URL
            }
        ]
    }
