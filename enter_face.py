import cv2
from resources.face_recognition import detect_one
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Name of person to take a picture of')

cam = cv2.VideoCapture(0)

cv2.namedWindow("Your Face!")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] failed to grab frame")
        break
    cv2.imshow("Your Face!", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("[INFO] Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        args = parser.parse_args()
        img = detect_one(frame)

        if img is None:
            print("[ERROR] Image isn't clear! Try again!")
            continue
        dir = 'data/faces/'+args.name+'/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(dir+str(len(os.listdir(dir)))+'.npy','wb') as f:
            np.save(f,img)
        print('[INFO] {} Saved Successfully'.format(args.name))


