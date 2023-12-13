import cv2
import numpy

# 이미지 불러오기
origin = cv2.imread('./a.jpg')
img = cv2.imread('./a.jpg')
height, width, channel = img.shape
print('original image shape:', height, width, channel)

# 인식에 사용할 이미지 크기 재조정
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
print('blob shape:', blob.shape)

# coco 이름 파일 불러오기
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print('number of classes =', len(classes))

# configuration 파일, weight 파일 불러오기
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# 출력 레이어 설정
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('output layers:', output_layers)

# 사물 인식
net.setInput(blob)
outs = net.forward(output_layers)

# 사물의 테두리와 신뢰값 설정
class_ids = []
confidence_scores = []
boxes = []

for out in outs:

    for detection in out:

        scores = detection[5:]
        class_id = numpy.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)

print('number of detected objects =', len(boxes))

# 신뢰도 낮은 사물 테두리 제거
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
print('number of final objects =', len(indices))

# 이미지에 사물 테두리와 라벨 표시
colors = numpy.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN

# 다리 수를 세는 변수
total_legs = 0

for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(f'class {label} detected at {x}, {y}, {w}, {h}')
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

        # 사람이 감지되었을 때 2만큼 증가
        if label.lower() == 'person':
            total_legs += 2
        # 동물이 감지되었을 때 4만큼 증가 (list 업데이트 필요)
        if label.lower() in ['cat', 'dog', 'horse']:
            total_legs += 4

# 원본 이미지 표시
cv2.imshow('Original Image', origin)
cv2.waitKey()
cv2.destroyAllWindows()

# 다리 수 물어보기
guess = int(input('\nGuess the total number of legs in the picture: '))
if guess == total_legs:
    print('Correct!')
else:
    print('Incorrect! Total number of legs in the picture:', total_legs)

# 처리된 이미지 표시
cv2.imshow('Processed Image', img)
cv2.waitKey()
cv2.destroyAllWindows()
