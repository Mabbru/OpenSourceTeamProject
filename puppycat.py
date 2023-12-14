import cv2
import numpy as np
from google.colab import drive
from google.colab.patches import cv2_imshow  # cv2_imshow를 사용해 이미지를 표시합니다.

# Google Drive 마운트
drive.mount('/content/drive')

# 이미지 데이터셋 경로 설정
dataset_path = "/content/drive/My Drive/puppyAndcat/firstpic.jpeg"

# YOLO 모델과 설정 파일 로드
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# 클래스 이름 로드 (COCO 데이터셋 클래스)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# 이미지 로드
image_path = '/content/drive/My Drive/puppyAndcat/firstpic.jpeg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# YOLO 입력 설정
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
model.setInput(blob)

# 객체 검출
layer_names = model.getLayerNames()
unconnected_out_layers = model.getUnconnectedOutLayers()

# OpenCV 버전에 따라 반환 값 형식이 다를 수 있음을 고려하여 처리
if unconnected_out_layers.ndim == 2:  # 리스트의 리스트 형태인 경우
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
else:  # 단일 리스트 형태인 경우
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

outs = model.forward(output_layers)

# 검출된 객체에 바운더리 박스 추가
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # 바운더리 박스 계산
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # 사각형 그리기
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 클래스 이름과 신뢰도 추가
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 표시
cv2_imshow(image)
