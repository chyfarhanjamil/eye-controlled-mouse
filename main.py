import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
print(pyautogui.size())

while cam.isOpened():
    # Read a frame from the video stream
    ret, frame = cam.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB format for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks in the RGB frame
    results = face_mesh.process(rgb_frame)
    #print(landmark_points)

    frame_h, frame_w, _ = frame.shape
    #print(frame.shape)

    # Extract the landmarks of the first detected face
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        for id, landmark in enumerate(landmarks[474:478]):
            # Get the position of the left eye
            left_eye_x = int(landmark.x * frame.shape[1])
            left_eye_y = int(landmark.y * frame.shape[0])
            #print(x, y)

            # Draw circles at the landmarks on the frame for visualization for landmark in landmarks:
            cv2.circle(frame, (left_eye_x, left_eye_y), 3, (0, 0, 255))

            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y

                # Move the mouse cursor to the position of the left eye
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]

        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
        if (left[0].y - left[1].y) < 0.01:
            pyautogui.click()
            pyautogui.sleep(1)
    cv2.imshow('Eye Gaze Tracker', frame)
    cv2.waitKey(1)