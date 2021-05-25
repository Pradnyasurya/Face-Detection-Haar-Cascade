import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')          #Face will be detected using this model
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')   #Eye will be detected using this model

# Read the input image
#img = cv2.imread('test.png')                           #For an image
cap = cv2.VideoCapture('selfie_video.mp4')              #For a video stored in the directory
#cap = cv2.VideoCapture(0)                              #For live video captured by the webcamera 

while cap.isOpened():
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        face_1 = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_1)
        colors = np.random.randint(1, 255, 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (int(colors[0]), int(colors[1]), int(colors[2])), thickness=2)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
        
        for (x2, y2, w2, h2) in eyes:
            eye_colors = np.random.randint(1, 255, 3)
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            eye_radius = int(round((w2 + h2) * 0.25))
            cv2.circle(img, center=eye_center, radius=eye_radius,
                       color=(int(eye_colors[0]), int(eye_colors[1]), int(eye_colors[2])))
            
    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()