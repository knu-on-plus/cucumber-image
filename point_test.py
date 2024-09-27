import matplotlib.pyplot as plt
import cv2

# 전역 변수로 사용할 포인트 리스트
input_points = []

# 마우스 이벤트 콜백 함수
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        input_points.append([x, y])  # 클릭한 좌표를 리스트에 추가
        print(f"Point selected: ({x}, {y})")
        plt.plot(x, y, 'ro')  # 선택한 포인트를 빨간 점으로 표시
        plt.draw()  # 그래프 업데이트

# 이미지 로드 (여기에 이미지를 불러올 경로를 설정하세요)
image_path = "/home/oi/Desktop/song/cucumber-image/data/images/V003_3_3_1_2_4_2_2_1_0_0_20221019_5319_20240422195059.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 이미지를 RGB로 변환

# 이미지 표시 및 클릭 이벤트 연결
plt.ion()  # 인터랙티브 모드 활성화
fig, ax = plt.subplots()
ax.imshow(image_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# 창을 닫은 후 선택된 모든 포인트 출력
print("All selected points:", input_points)
