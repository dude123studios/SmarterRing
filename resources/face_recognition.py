from models.face_recognition200_model import model
import cv2
import os
import numpy as np
import time
from resources.detection import detect_face


def detect_one(cv2_img):
    return preprocess(detect_face(cv2_img)[0])


def preprocess(cv2_img):
    cv2_img = cv2.resize(cv2_img,(64,64))
    cv2_img = cv2_img/127.5-1
    return cv2_img


def detect_all(cv2_img):
    imgs = detect_face(cv2_img)
    if imgs is None:
        return None
    imgs = [preprocess(img) for img in imgs]
    return imgs


def is_same_person(imgA, imgB):
    imgA = np.expand_dims(imgA, axis=0)
    imgB = np.expand_dims(imgB, axis=0)
    input_ = {'inputA': imgA, 'inputB': imgB}
    out = model(input_)
    return out.numpy().tolist()[0][0]


def who_is_here(img):
    start = time.time()
    people = detect_all(img)
    print('finished image detection in:', str(time.time() - start))
    best_people = []
    strengths = []
    if people is None:
        return None
    for person in people:
        best = 1
        best_name = ''
        for person_path in os.listdir('data/faces/'):
            for image_path in os.listdir('data/faces/'+person_path):
                img = np.load('data/faces/' + image_path, allow_pickle=True)
                val = is_same_person(person, img)
                if val < best:
                    best = val
                    best_name = image_path.split('.')[0]
        if best > 0.4:
            best_name = 'UNKNOWN'
        best_people.append(best_name)
        strengths.append(best)
    print(strengths)
    return best_people
