# -*- coding: utf-8 -*-

'''Working script'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
# import cv2

# List of emotions 
emotion =  ['Anger', 'Happy', 'Neutral', 'Surprise', 'Sad']

# loads model
model = keras.models.load_model("my_model.h5")

# Select font 
# font = cv2.FONT_HERSHEY_SIMPLEX

# Call the camera device=0.
cam = cv2.VideoCapture(0)

# Loads Harr Cascade file used to detect faces. 
face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# While Loop for inference 'while'  camera is on.
while True:
    ret, frame = cam.read()
    
    if ret==True:
        # turn color into B/W images.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # recongnise face.
        faces = face_cas.detectMultiScale(gray, 1.3,5)
        
        for (x, y, w, h) in faces:
            face_component = gray[y:y+h, x:x+w]
            fc = cv2.resize(face_component, (48, 48))
            inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
            inp = inp/255.
            prediction = model.predict(inp)
            em = emotion[np.argmax(prediction)]
            score = np.max(prediction)
            
            # create square and percentage.
            cv2.putText(frame, em+"  "+str(score*100)+'%', (x, y), font, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("image", frame)
        
        if cv2.waitKey(1) == 27:
            break
    else:
        print ('Error')

# Close Camera.
cam.release()
cv2.destroyAllWindows()
