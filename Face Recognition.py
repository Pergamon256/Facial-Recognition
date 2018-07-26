import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/pranav/Documents/Main/Projects/FacialRecognition/trainer/trainer.yml')
cascadePath = "/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
try:
	with open('/Users/pranav/Documents/Main/Projects/FacialRecognition/names.txt') as f:
		names = [line.rstrip() for line in f]
except FileNotFoundError:
	names = []

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

#Functions below define minimum window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.2,
		minNeighbors = 5,
		minSize = (int(minW), int(minH))
	)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

		#Functions below check if confidence is less than 100, and 0% confidence is a perfect match
		if (confidence < 100):
			id = names[id]
			confidence = "  {0}%".format(round(100 - confidence))
		else:
			id = "Unrecognized"
			confidence = "  {0}%".format(round(100 - confidence))

		cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
		cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,255), 1)

	cv2.imshow('camera', img)
	k = cv2.waitKey(10) & 0xFF
	if k == 27: # press ESC to quit
		break

print("\n [INFO] Exiting program and cleaning up.")
cam.release()
cv2.destroyAllWindows()