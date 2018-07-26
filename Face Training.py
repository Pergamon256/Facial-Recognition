import cv2, os, glob
import numpy as np
from PIL import Image

os.chdir("/Users/pranav/Documents/Main/Projects/FacialRecognition/dataset2/")
try:
	os.remove('.DS_Store')
except OSError:
	pass
os.chdir("/Users/pranav")

path = '/Users/pranav/Documents/Main/Projects/FacialRecognition/dataset2'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faceSamples = []
	ids = []
	for imagePath in imagePaths:
		PIL_img = Image.open(imagePath).convert('L') #convert to grayscale
		img_numpy = np.array(PIL_img, 'uint8')
		
		id = int(os.path.split(imagePath)[-1].split(".")[1])
		faces = detector.detectMultiScale(img_numpy)
		
		for (x,y,w,h) in faces:
			faceSamples.append(img_numpy[y:y+h,x:x+w])
			ids.append(id)
	
	return faceSamples,ids

print("\n [INFO] Training faces. This will take a few seconds. Please wait.")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.save('/Users/pranav/Documents/Main/Projects/FacialRecognition/trainer/trainer.yml') #recognizer.save() may not work in linux/pi

print("\n [INFO] {0} faces trained. The program will now end.".format(len(np.unique(ids))))