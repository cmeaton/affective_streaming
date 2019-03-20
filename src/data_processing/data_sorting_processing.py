import cv2
import glob
import tensorflow as tf
from shutil import copyfile
import os
import pandas as pd
import csv

def organize_data():
    '''This function sorts the downloaded folder structure so that a subdirectory for each emotion is populated
    with their corresponding images.'''

    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
    participants = glob.glob("source_emotion//*") #Returns a list of all folders with participant numbers
    for x in participants:
        part = "%s" %x[-4:] #store current participant number
        for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
            for files in glob.glob("%s//*" %sessions):
                current_session = files[20:-30]

                with open(files, 'r') as f:
                    file = f.read()
                emotion = int(float(file)) #emotions are encoded as a float, readline as float, then convert to integer.
                sourcefile_emotion = sorted(glob.glob("source_images/%s/%s/*" %(part, current_session)))[-1] #get path for last image in sequence, which contains the emotion
                sourcefile_neutral = sorted(glob.glob("source_images/%s/%s/*" %(part, current_session)))[0] #do same for neutral image
                dest_neut = "sorted_set//neutral//%s" %sourcefile_neutral[25:] #Generate path to put neutral image
                dest_emot = "sorted_set//%s//%s" %(emotions[emotion], sourcefile_emotion[25:]) #Do same for emotion containing image
                copyfile(sourcefile_neutral, dest_neut) #Copy file
                copyfile(sourcefile_emotion, dest_emot) #Copy file

organize_data()

faceDet = cv2.CascadeClassifier('/Users/cmeaton/Documents/code/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
faceDet_two = cv2.CascadeClassifier("/Users/cmeaton/Documents/code/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("/Users/cmeaton/Documents/code/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("/Users/cmeaton/Documents/code/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml")



def detect_faces(emotion):
    
    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
    files = glob.glob("sorted_set/%s/*" %emotion) #Get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
#         img = tf.image.grayscale_to_rgb(frame)
       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        # gray = cv2.cvtColor(frame, cv2.CV_GRAY2RGB)
        #img = cv2.cvtColor(frame, CV_GRAY2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale

        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file: %s" %f)
#             gray = gray[y:y+h, x:x+w] #Cut the frame to size
            gray = gray[y:y+h, x:x+w] #Cut the frame to size

            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("data_set/%s/%s.jpg" %(emotion, filenumber), out) #Write image
            except:
                pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions:
    detect_faces(emotion) 


# In[ ]:




