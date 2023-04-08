
# import necessary libraries
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# set the path to the folder containing the images to be used for recognition
path = 'ImagesAttendance'


# create empty lists to store the images and their corresponding class names
images = []
classNames = []


# get a list of all the files in the specified folder
mylist = os.listdir(path)
print(mylist)


# loop through each file in the folder
for cl in mylist:
    # read the image file using OpenCV and add it to the images list
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # extract the class name from the file name and add it to the classNames list
    classNames.append(os.path.splitext(cl)[0])


# print the list of images and their class names
#print(images)
#print(classNames)


# define a function to encode the face features in a given image
def find_encodings(images):
    # create an empty list to store the face encodings
    encode_list = []
    # loop through each image in the images list
    for img in images:
        # convert the image from BGR (OpenCV's default color format) to RGB (the format expected by the face_recognition library)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # use the face_recognition library to generate a list of face encodings in the image (we assume there is only one face per image)
        encode = face_recognition.face_encodings(img)[0]
        # add the encoding to the encode_list
        encode_list.append(encode)
        # return the list of face encodings
    return encode_list


# define a function to mark attendance for a given person
def mark_attendance(name):
    # open the attendance file in read-write mode
    with open('CEO_Attendance.csv', 'r+') as f:
        # read all the lines from the file and store them in a list
        my_data_list = f.readlines()
        # create an empty list to store the names already present in the file
        name_list = []

        # loop through each line in the file
        for line in my_data_list:
            # split the line by comma and extract the first element (the name) and add it to the name_list
            entry = line.split(',')
            name_list.append(entry[0])

        # if the name is not already in the name_list
        if name not in name_list:
            # get the current date and time
            now = datetime.now()
            dtstring = now.strftime("%m/%d/%Y, %H:%M:%S")
            # write the name and the current date/time to the file (in CSV format)
            f.writelines(f'\n{name},{dtstring}')


# encode the face features in the images
encodeknown = find_encodings(images)
# print a message to indicate that the encoding is complete
print('encoding complete')


# create a new video capture object using the default camera
cap = cv2.VideoCapture(0)


# start an infinite loop to process each frame of the video stream
while True:
    # read a frame from the video stream
    success, img = cap.read()
    # resize the image to a smaller size (to speed up processing)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # convert the image from BGR to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all faces in the current frame
    faceCurrentFrame = face_recognition.face_locations(imgS)
    # Encode the faces in the current frame
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    # Loop through each face and try to recognize it
    for encodeface, faceloc in zip(encodeCurrentFrame, faceCurrentFrame):
        # Compare the current face to the known faces and get a list of matches
        matches = face_recognition.compare_faces(encodeknown, encodeface)
        # Calculate the distance between the current face and each known face
        face_dis = face_recognition.face_distance(encodeknown, encodeface)
        print(face_dis)
        # Find the index of the known face with the smallest distance to the current face
        matchIndex = np.argmin(face_dis)

        # If there is a match, mark attendance and display the name on the screen
        if matches[matchIndex]:
            # get the name of the matched person and convert to uppercase
            name = classNames[matchIndex].upper()
            print(name)
            # get the coordinates of the detected face in the current frame
            y1, x2, y2, x1 = faceloc
            # scale the coordinates up by a factor of 4 to match the original frame size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # draw a green rectangle around the detected face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw a green rectangle at the bottom of the face rectangle to act as a background for the name text
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            # write the name of the detected person on the frame
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # mark the attendance of the detected person in the CSV file
            mark_attendance(name)

    # show the current frame with detected faces and names
    cv2.imshow('webcam', img)

    # wait for a key press and check if it's the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # exit the loop if the 'q' key is pressed
        break

