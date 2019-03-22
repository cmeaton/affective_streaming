import cv2
import numpy as np
from imutils import face_utils
import math
import dlib
from sklearn.svm import SVC
import pickle

# load model
filename = 'models/saved_models/lin_svm_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# load facial landmark algorithms
p = 'models/saved_models/face_algorithms/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#load facial finder algorithms
faceDet = cv2.CascadeClassifier(
    'models/saved_models/face_algorithms/haarcascade_frontalface_default.xml'
)
faceDet_two = cv2.CascadeClassifier(
    "models/saved_models/face_algorithms//haarcascade_frontalface_alt2.xml"
)
faceDet_three = cv2.CascadeClassifier(
    "models/saved_models/face_algorithms//haarcascade_frontalface_alt.xml"
)
faceDet_four = cv2.CascadeClassifier(
    "models/saved_models/face_algorithms//haarcascade_frontalface_alt_tree.xml"
)

# placeholder for data to be used later
data = {}
# normalization of image arrays
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def get_landmarks(image):
    '''This function locates facial landmarks and computes the relative distance from the mean for each point.'''


    training_data = []
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

    training_data.append(data['landmarks_vectorised'])
    return training_data

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


def convert_img_bytes_to_numpy_array(img_bytes):
    """
    Convert a buffer of bytes into a numpy array usable for processing.
    """
    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)


def convert_numpy_array_to_img(numpy_array):
    """
    Convert processed numpy array into image bytes.

    docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imencode
    """
    _, encoded_image = cv2.imencode('.jpeg', numpy_array)
    return encoded_image.tobytes()

def process_frame(frame):
    """
    Take an image, process it, return edited image.
    """
    # Capture frame-by-frame
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

        results = (get_landmarks(face_img))
        model = loaded_model.predict_proba(results)
        if model[0][0] > .3:
            emotion = 'Anger'
            fontColor = (255, 0, 0)
            font = cv2.FONT_ITALIC
            fontScale = 1.5
        elif model[0][1] > .3:
            emotion = 'Disgust'
            fontColor = (60, 179, 113)
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 1.5
        elif model[0][2] > .5:
            emotion = 'Fear'
            fontColor = (255, 0, 0)
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 1
        elif model[0][3] > .5:
            emotion = 'Happy'
            fontColor = (0, 221, 221)
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            fontScale = 2
        elif model[0][4] > .4:
            emotion = 'Neutral'
            fontColor = (255,255,255)
            font = cv2.FONT_ITALIC
            fontScale = 1.5
        elif model[0][5] > .3:
            emotion = 'Surprise'
            fontColor = (196, 0, 255)
            font = cv2.FONT_HERSHEY_TRIPLEX
            fontScale = 1.5
        else:
            emotion = ''
            fontColor = (255,255,255)
            font = cv2.FONT_ITALIC
            fontScale = 1.5

        bottomLeftCornerOfText = (10,100)
        lineType               = 2
        cv2.putText(frame, f'Predicted emotion: {emotion}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
    # Get faces into webcam's image
    rects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Draw on our image, all the finded cordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    return frame


def run_video():
    # The code below creates a live video stream from webcam
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        process_frame(frame)
        cv2.imshow('Keras Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_video()
