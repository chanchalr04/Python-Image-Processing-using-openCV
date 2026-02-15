import cv2
import cv2.data
import os


# print(os.listdir(cv2.data.haarcascades))
training_file ="haarcascade_frontalface_default.xml"
file_path = cv2.data.haarcascades + training_file
print(file_path)

model = cv2.CascadeClassifier(file_path)
img = cv2.imread("image2.jpg")

faces = model.detectMultiScale(img,1.3,5)
for face in faces:
   x1 = face[0]
   y1 = face[1]
   x2 = x1 + face[2]
   y2 = y1 + face[3]
   cv2.rectangle(img,(x1,y1),(x2,y2),[255,0,0])
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ====> this is use to change color into gray
new_img = cv2.resize(img,(400,400)) # ==> img(width,height)
cv2.imshow("group image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()