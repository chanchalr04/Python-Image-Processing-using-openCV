import cv2  #  pip install opencv-python
import cv2.data
import os


# print(os.listdir(cv2.data.haarcascades))
training_file ="haarcascade_frontalface_default.xml"
file_path = cv2.data.haarcascades + training_file

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier(file_path)

while True:
   status,img = cam.read()
   if status == False:
      print("camera is not open")
      break
   faces = model.detectMultiScale(img,1.3,5)
   for face in faces:
      x1 = face[0]
      y1 = face[1]
      x2 = x1+face[2]
      y2 = y1 + face[3]
      cv2.rectangle(img,(x1,y1),(x2,y2),[0,0,255],2)
   cv2.imshow("Video",img)
   key=cv2.waitKey(1)
   if key==ord("c"):
      cv2.destroyAllWindows()
      break


