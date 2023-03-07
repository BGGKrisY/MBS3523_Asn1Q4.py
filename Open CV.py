import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Frame = cv2.VideoCapture(0)

while True:
    ret, frame = Frame.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mask = cv2.rectangle(np.zeros_like(frame), (x, y), (x+w, y+h), (255, 255, 255), -1)
        masked_frame = cv2.bitwise_or(cv2.bitwise_and(frame, mask), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cv2.putText(masked_frame, 'MBS3523 Assignment 1 – Q3 Name: YEUNG YAT', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (125, 0, 125), 2)
    cv2.imshow('Face Tracking', masked_frame)
    if cv2.waitKey(1) & 0xFF == ord('o'):
        break
Frame.release()
cv2.destroyAllWindows()