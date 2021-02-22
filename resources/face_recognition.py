import tensorflow as tf
from models.face_recognition_model import model
import cv2
import os
import pickle

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def preprocess_img(image_path):
    img = cv2.imread(image_path)
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
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    img = img[y:y + h, x:x + w]
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, (64, 64))
    # use mobilenet preprocessing for convenience, even if we use a smaller model
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def just_detect(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return None
    # produce image crops of just the face of a person
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    imgs = [img[y:y + h, x:x + w] for (x, y, w, h) in faces]
    imgs = [tf.convert_to_tensor(img, dtype=tf.float32) for img in imgs]
    imgs = [tf.image.resize(img, (64, 64)) for img in imgs]
    # use mobilenet preproccesing for convinience, even if we use a smaller model
    imgs = [tf.keras.applications.mobilenet_v2.preprocess_input(img) for img in imgs]
    return imgs

def is_same_person(imgA, imgB):
    imgA = tf.expand_dims(imgA, axis=0)
    imgB = tf.expand_dims(imgB, axis=0)
    input_ = {'inputA': imgA, 'inputB': imgB}
    out = model(input_)
    return out.numpy().tolist()[0][0]


def who_is_here(img):
    people = just_detect(img)
    if people is None:
        # people may have left the frame or turned around
        return None
    best_people = []
    strengths = []
    for person in people:
        best = 1
        best_name = ''
        for image_path in os.listdir('data/faces/'):
            preprocessed = preprocess_img('data/faces/' + image_path)
            if preprocessed is not None:
                val = is_same_person(person, preprocess_img('data/faces/' + image_path))
            else:
                raise LookupError("Face: {} in the directory data/faces/ ".format(person) +
                                  "may not contain a face, or isn't very clear")
            if val < best:
                best = val
                best_name = image_path.split('.')[0]
        if best > 0.4:
            best_name = 'UNKNOWN'
        best_people.append(best_name)
        strengths.append(best)
    print(strengths)
    return best_people


