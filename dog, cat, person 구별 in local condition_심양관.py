!pip install opencv-python
!pip install opencv-python-headless

import cv2
import numpy as np

# YOLO 모델 로드
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# 클래스 이름 로드
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# YOLO 설정
layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

from google.colab.patches import cv2_imshow

def detect_objects(image):
    height, width, _ = image.shape

    # 이미지 전처리 및 모델에 전달
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # 감지된 객체 정보 저장
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과 출력
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# 이미지 읽기
image_path = './image name.jpg' # 이미지 주소 입력
image = cv2.imread(image_path)

# 객체 감지
result_image = detect_objects(image)

# 결과 이미지 출력
cv2_imshow(result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
