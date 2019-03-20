from keras.preprocessing.image import load_img
from keras.models import load_model
import cv2
import numpy as np

model_path = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/best.h5'
model = load_model(model_path)
casc_path = '/Users/cmeaton/Documents/code/opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(casc_path)

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

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(224, 224),
   )


    face_imgs = np.empty((len(faces), 224, 224, 3))

    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, margin=40, size=224)
        (x, y, w, h) = cropped
        cv2.rectangle(frame, (x, y), (x + w, y + h), (224, 224, 3), 2)
        face_imgs[i,:,:,:] = face_img
    if len(face_imgs) > 0:
        # predict ages and genders of the detected faces
        results = model.predict(face_imgs)


        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(frame, f'{results}',
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)


#    cv2.imshow('Keras Faces', frame)
    cv2.imshow('Keras Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
