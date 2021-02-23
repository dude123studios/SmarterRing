import cv2
from resources.face_recognition import detect_one
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Name of person to take a picture of')

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        args = parser.parse_args()
        img = detect_one(frame)
        with open('data/faces/'+args.name+'.npy','wb') as f:
            np.save(f, img)
        print('{} Saved Successfully'.format(args.name))
        break