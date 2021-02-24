import tensorflow as tf
from models.face_recognition_model import model
import cv2
import os
import pickle
import numpy as np
import time

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def detect_one(img):
    # some images wont work
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None
    # produce image crops of just the face of a person
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(120, 120),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[-1]
    img = img[y:y + h, x:x + w]
    return img

def preprocess(cv2_img):
    img = tf.convert_to_tensor(cv2_img, dtype=tf.float32)
    img = tf.image.resize(img, (64, 64))
    # use mobilenet preprocessing for convenience, even if we use a smaller model
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def detect_all(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None
    # produce image crops of just the face of a person
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    imgs = [img[y:y + h, x:x + w] for (x, y, w, h) in faces]
    imgs = [preprocess(img) for img in imgs]
    return imgs

def is_same_person(imgA, imgB):
    imgA = tf.expand_dims(imgA, axis=0)
    imgB = tf.expand_dims(imgB, axis=0)
    input_ = {'inputA': imgA, 'inputB': imgB}
    out = model(input_)
    return out.numpy().tolist()[0][0]


def who_is_here(img):
    start = time.time()
    people = detect_all(img)
    print('finished image detection in:', str(time.time() - start))
    best_people = []
    strengths = []
    if people == None:
        return None
    for person in people:
        best = 1
        best_name = ''
        for image_path in os.listdir('data/faces/'):
            img = np.load('data/faces/'+image_path,allow_pickle=True)
            img = preprocess(img)
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


