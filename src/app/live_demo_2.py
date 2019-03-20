
import cv2
import numpy as np

from imutils import face_utils
import dlib





p = '/Users/cmeaton/Documents/code/ds/METIS/sea19_ds7_workingdir/project_5/src/models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)




video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(frame, f'facial landmarks',
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
