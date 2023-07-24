# Face Recognition Attendance System using OpenCV and Face_Recognition Library

This program recognizes faces from the live video feed and marks the attendance of recognized faces in a CSV file. The face recognition algorithm is implemented using the *'face_recognition'* library in Python.

## Requirements
* Python 3.x
* OpenCV (pip install opencv-python)
* Face_Recognition (pip install face-recognition)
* numpy (pip install numpy)

## Usage
1. Store images of people to be recognized in a folder named 'ImagesAttendance'.
2. Run the program. It opens the webcam and starts recognizing faces.
3. When a recognized face is detected, the program marks the attendance of that person in a CSV file named 'CEO_Attendance.csv'.

## Functionality
* find_encodings(images): This function takes a list of images as input and returns the facial encodings of each image. It uses the face_recognition library to detect facial features and generate the encodings.
* mark_attendance(name): This function marks the attendance of the person whose face is recognised. It reads from and writes to a CSV file named 'CEO_Attendance.csv'. If the person is not already marked present in the CSV file, it appends a new line with the person's name and current timestamp.

## How it works
1. The program reads the images from the 'ImagesAttendance' folder and generates facial encodings for each image using the find_encodings() function.
2. It then opens the webcam and starts reading frames.
3. For each frame, the program detects the location of all faces using the face_recognition library.
4. It then compares the facial encodings of all detected faces with the encodings generated earlier from the images in the 'ImagesAttendance' folder using the compare_faces() function.
5. If a match is found, the person's name is retrieved from the list of image file names using the classNames list.
6. A rectangle is drawn around the detected face, and the person's name is displayed above the rectangle.
7. The mark_attendance() function is called to mark the attendance of the recognized person in the 'CEO_Attendance.csv' file.

### Note:
The accuracy of the face recognition algorithm depends on the quality of the images stored in the 'ImagesAttendance' folder and the lighting conditions during recognition.





