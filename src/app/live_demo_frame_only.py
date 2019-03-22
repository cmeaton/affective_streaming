import cv2
import numpy as np
from imutils import face_utils
import glob
import random
import math
import dlib
import itertools
from sklearn.svm import SVC
import pickle

# load model
filename = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/lin_svm_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# load facial landmark algorithms
p = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#load facial finder algorithms
faceDet = cv2.CascadeClassifier(
    '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms/haarcascade_frontalface_default.xml'
)
faceDet_two = cv2.CascadeClassifier(
    "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms//haarcascade_frontalface_alt2.xml"
)
faceDet_three = cv2.CascadeClassifier(
    "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms//haarcascade_frontalface_alt.xml"
)
faceDet_four = cv2.CascadeClassifier(
    "/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/saved_models/face_algorithms//haarcascade_frontalface_alt_tree.xml"
)

# placeholder for data to be used later
data = {}
# normalization of image arrays
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def crop_face(imgarray, section, margin=40, size=64):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w,h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w-1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h-1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


# The code below creates a live video stream from webcam

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDet_three.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(224, 224),
   )

    face_imgs = np.empty((len(faces), 350, 350, 3))

    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, margin=40, size=350)
        (x, y, w, h) = cropped
        cv2.rectangle(frame, (x, y), (x + w, y + h), (350, 350, 3), 2)
        face_imgs[i,:,:,:] = face_img


    cv2.imshow('Keras Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
