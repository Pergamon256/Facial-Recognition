import cv2, os, shutil

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

#Detect face and assign to an ID
face_detector = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

face_id = input('\n Enter user ID: ')
os.makedirs("/Users/pranav/Documents/Main/Projects/FacialRecognition/dataset/" + str(face_id), exist_ok=False)

#Add names to text file to be read by Face Recognition script for 'names' list
name = input('\n Please enter your name: ')
with open('/Users/pranav/Documents/Main/Projects/FacialRecognition/names.txt', 'w') as f:
	f.write(name + '\n')

print("\n [INFO] Initializing face capture. Look at the camera and wait.")

count = 0
while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(
		gray,
		scaleFactor = 1.3,
		minNeighbors = 5
	)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		count += 1

		os.chdir("/Users/pranav/Documents/Main/Projects/FacialRecognition/dataset/" + str(face_id))

		cv2.imwrite("User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
		cv2.imshow('image', img)

		os.chdir("/Users/pranav")

	k = cv2.waitKey(100) & 0xFF
	if k == 27: # press 'ESC' to quit
		break
	elif count >= 30:
		break

#Copy images to 'dataset2' directory
src = r'/Users/pranav/Documents/Main/Projects/FacialRecognition/dataset'
dest = r'/Users/pranav/Documents/Main/Projects/FacialRecognition/dataset2'

for path, subdirs, files in os.walk(src):
	for name in files:
		filename = os.path.join(path, name)
		shutil.copy2(filename, dest)

print("\n [INFO] Exiting program and cleaning up.")
cam.release()
cv2.destroyAllWindows()