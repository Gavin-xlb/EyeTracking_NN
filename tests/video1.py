import face_recognition
import cv2
import math

# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    if len(face_locations) != 0:
        # 找到最大的人脸作为检测人脸
        max_area = 0
        max_index = 0
        for i in range(len(face_locations)):
            top, right, bottom, left = face_locations[i]
            if math.fabs((top - bottom) * (right - left)) > max_area:
                max_area = math.fabs((top - bottom) * (right - left))
                max_index = i
        top, right, bottom, left = face_locations[max_index]
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        # Extract the region of the image that contains the face
        face_image = frame[top:bottom, left:right]
        face_landmarks_list = face_recognition.face_landmarks(face_image)
        for face_landmarks in face_landmarks_list:
            # Let's trace out each facial feature in the image with a line!
            for facial_feature in face_landmarks.keys():
                # if facial_feature == 'right_eye':
                    # print(facial_feature, face_landmarks[facial_feature])
                for point in face_landmarks[facial_feature]:
                    cv2.circle(face_image, point, 0, (0, 0, 255), 3)

        # Put the blurred face region back into the frame image

        frame[top:bottom, left:right] = face_image

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
