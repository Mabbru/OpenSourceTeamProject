import cv2

dataset_path = "/Users/choeseongjun/Downloads/puppyAndcat/firstpic.jpeg"

# 이미지 로드
image = cv2.imread(dataset_path)

# Haar Cascade Classifier 로드 (예: 얼굴 검출)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지를 흑백으로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# 검출된 얼굴 주위에 텍스트 추가
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText 함수에서 'is_cat' 함수는 정의되어 있지 않아 주석 처리합니다.
    # cv2.putText(image, "고양이" if is_cat(x, y, w, h) else "강아지", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 표시
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
