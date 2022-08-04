# -*- coding: utf-8 -*-

'''Working script'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import emoji
from PIL import Image,ImageDraw,ImageFont

boob1 = "\U0001f600"
boob2 = "\U0001f600"
boob3 = "\U0001f600"
boob4 = "\U0001f600"
boob5 = "\U0001f600"
boob6 = "\U0001f600"
boob7 = "\U0001f600"
# List sed to map the labels ['0', '1', '2', '3', '4', '5', '6', '7'] to their plain text descriptors. in line: 48. 
# emotion =  [boob1, boob2, boob3, boob4, boob5, boob6, boob7]
emotion = ["\U0001f600", "\U0001f600", "\U0001f600", "\U0001f600", "\U0001f600", "\U0001f600", "\U0001f600"]

# loads the model .h5 file.
model = keras.models.load_model("my_model2.h5")

# Defines which font to use.
font = ImageFont.truetype("arial",32)

# Defines where to get input from in this case being the camera device=0.
cam = cv2.VideoCapture(0)

# Loads Harr Cascade file used to detect faces. 
face_cas = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
draw = ImageDraw.Draw(emotion)
# Loop runs inference 'while'  camera is working/on.
while True:
    ret, frame = cam.read()
    
    if ret==True:
        # turns color into B/W images.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # uses the haar cascades/haarcascade_frontalface_default.xml file to determine if there's a face from the input.
        faces = face_cas.detectMultiScale(gray, 1.3,5)
        
        # Takes in the input resizes it and reshapes it to 48x48, as our model was trained on 48x48 input, 
        # turns it into a float and divides by 225 (255 is our output resolution), array inp after being divided by 255 
        # is used as contant in variable: prediction that predicts what the input expression may be using the model.
        for (x, y, w, h) in faces:
            face_component = gray[y:y+h, x:x+w]
            fc = cv2.resize(face_component, (48, 48))
            inp = np.reshape(fc,(1,48,48,1)).astype(np.float32)
            inp = inp/255.
            prediction = model.predict(inp)
            em = emotion[np.argmax(prediction)]
            score = np.max(prediction)
            
            # creates the triangle that show the predicted emotion around the users face on the output window, and creates the output frame.
            drawText(frame, em+"  "+str(score*100)+'%', (x, y), font, 1, (0, 255, 0), 2)
            # cv2.putText(frame, em+"  "+str(score*100)+'%', (x, y), font, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow("image", frame)
        
        if cv2.waitKey(1) == 27:
            break
    else:
        print ('Error')

# Meant to release the camera from usage if window turns off or the while loop throws back "Error" however doesn't work in this version of OpenCV.
cam.release()
cv2.destroyAllWindows()
