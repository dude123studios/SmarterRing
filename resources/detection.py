import cv2
import mtcnn

face_detector = mtcnn.MTCNN()
conf_t = 0.99

def detect_face(cv2_img):
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    faces = []
    for res in results:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        confidence = res['confidence']
        if confidence < conf_t:
            continue
        faces.append(cv2_img[x1:x2,y1:y2])
    return faces