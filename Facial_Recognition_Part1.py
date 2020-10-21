# {import libraries}
import cv2
import numpy as np

# face ke feature collected ho aisa system chahiye!! for this we use cascade classifier
# haarcascade ke andar body ke kisi n kisi part ko leke classification diye hue haii:
# face_classifier classifier ka object hai:

face_classifier = cv2.CascadeClassifier(
    'C:/Users/User/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#function for extracting the face feature
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB IMAGES KO GREY ME CHANGE KYUKI YE AASAN PADEGA
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces: # CO-ORDINATES FOR FACES:
        cropped_face = img[y:y + h, x:x + w] # LIST OF IMAGES

    return cropped_face

#for camera
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:  #FRAME IS CAMERA IMAGES:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))  # CAMERA SIZE RESIZING {FRAME ME PEHLE IMAGE THEN RESIZING VALUE }
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)         # FACE IN COLOR TO GRAY

        file_name_path = 'C:/Users/User/Downloads/opencv_master/FACES/user' + str(count) + '.jpg'   # FOR SAVING THE IMAGES
        cv2.imwrite(file_name_path, face)   #IMAGES KO WRITE

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) # cv2. putText() method is used to draw a text string
        cv2.imshow('Face Cropper', face)

    else:
        print('Face not Found')
        pass

    # will display a frame for 1 ms, after which display will be automatically closed
    if cv2.waitKey(1) == 13 or count ==100:
        break

cap.release()                                      # CAMERA RELEASE
cv2.destroyAllWindows()
print("collecting samples complete!!!")