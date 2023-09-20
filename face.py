import cv2

# Print OpenCV version
# print("OpenCV Version:", cv2.__version__)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#load some pre-trained data on face frontals from opencv (haar cascade algorithm)

#choose an image to detect faces in
img = cv2.imread('img.jpg')

#must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ConVerTColor, in opencv its BGR not RGB


#detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) # Detects objects of different sizes in the input image. The detected objects are returned as list of rectangles
 

#draw rectangels around the faces:

# (x, y, w, h) = face_coordinates[0]
for (x, y, w, h) in face_coordinates:
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#cv2.rectangle(image, (x, y), (x+w, y+h), (b, g, r), thickness)


# print(face_coordinates)

#Name of the window 
cv2.imshow('This is the name of the window', img)

#closes the window on press of any key
cv2.waitKey()


