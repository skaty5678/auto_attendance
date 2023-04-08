import cv2 #import opencv library
import face_recognition #import face recognition library


# Load the 'Elon Musk.jpg' image and convert color from BGR to RGB
img_elon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)


# Load the 'Bill Gates.jpg' image and convert color from BGR to RGB
img_test = face_recognition.load_image_file('ImagesBasic/Bill Gates.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)


# Find the location of the face in the 'Elon Musk.jpg' image and generate a 128-dimensional face encoding
face_loc = face_recognition.face_locations(img_elon)[0]
encode_face = face_recognition.face_encodings(img_elon)[0]


# Draw a rectangle around the detected face in the 'Elon Musk.jpg' image
cv2.rectangle(img_elon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)


# Find the location of the face in the 'Bill Gates.jpg' image and generate a 128-dimensional face encoding
face_loc_test = face_recognition.face_locations(img_test)[0]
encode_face_test = face_recognition.face_encodings(img_test)[0]


# Draw a rectangle around the detected face in the 'Bill Gates.jpg' image
cv2.rectangle(img_test,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(255,0,255),2)


# Compare the face encodings of Elon Musk and Bill Gates and calculate the Euclidean distance between the face encodings
results = face_recognition.compare_faces([encode_face],encode_face_test)
face_dis = face_recognition.face_distance([encode_face],encode_face_test)


# Print the comparison results and face distance
print(results)
print(face_dis)


# Display the comparison results and face distance on the 'Bill Gates.jpg' image
cv2.putText(img_test,f'{results} {round(face_dis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


# Display the 'Elon Musk.jpg' and 'Bill Gates.jpg' images with rectangles drawn around the detected faces
cv2.imshow('elon musk',img_elon)
cv2.imshow('elon test',img_test)


# Wait indefinitely until a key is pressed
cv2.waitKey(0)
