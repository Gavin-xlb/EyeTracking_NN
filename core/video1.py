import face_recognition
import cv2

# This is a demo of blurring faces in video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)

# Initialize some variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="cnn")

    # Display the results
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        # Extract the region of the image that contains the face
        face_image = frame[top:bottom, left:right]
        face_landmarks_list = face_recognition.face_landmarks(face_image)
        for face_landmarks in face_landmarks_list:
            left_x_min, left_y_min = 500, 500
            left_x_max, left_y_max = 0, 0
            for x, y in face_landmarks['right_eye']:
                if x < left_x_min:
                    left_x_min = x
                elif x > left_x_max:
                    left_x_max = x
                if y < left_y_min:
                    left_y_min = y
                elif y > left_y_max:
                    left_y_max = y
            left_eye = face_image[left_y_min - 5:left_y_max + 5, left_x_min - 5:left_x_max + 5]
            left_eye = cv2.resize(left_eye, ((left_x_max - left_x_min + 10) * 5, (left_y_max - left_y_min + 10) * 5))
            EC = ((left_x_max - left_x_min + 10) * 5 / 2, (left_y_max - left_y_min + 10) * 5 / 2)
            gray = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray, 65, 255, 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.erode(thresh, kernel)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            #thresh_2 = cv2.drawContours(thresh, contours, -1, (0, 0, 255), 5)
            if len(contours) > 0:
                contours = sorted(contours, key=cv2.contourArea)
                cnt = contours[-1]
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    CG = (cx, cy)
                    print('ECCG: ', CG[0] - EC[0], CG[1] - EC[1])
                cv2.imshow('asd', thresh)
                # cv2.imshow("try", left_eye)
                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
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
