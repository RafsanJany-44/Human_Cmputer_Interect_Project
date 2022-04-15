
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from pynput.keyboard import Key, Controller

mp_drawing = mp.solutions.drawing_utils # taking drwaing utils
mp_hands = mp.solutions.hands
joint_list =[[8,5,0]]  # ladbmarks of 8,5,0

def draw_finger_angles(image, results, joint_list):   # this line is drwaing the finger line and the text in the monitor and calculating angle
    # Loop through hands 
    for hand in results.multi_hand_landmarks:  #building a loop for hand marks
        # Loop through joint sets
        for joint in joint_list: #travversing each segment of land marks
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi) #calculating angle for decision left or right


            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) #puting text into the monitor
    return image, angle

def get_label(index, hand, results):  #getting the level
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))

            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))

            output = text, coords

    return output


cap = cv2.VideoCapture(0) #for vedio capturing

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                    ) # drwaing each land mark and lines in hand

                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw angles to image from joint list
            image, angle = draw_finger_angles(image, results, joint_list) #calculating angel and drwain in image
            keyboard = Controller() #calling keayboard controler
            if angle<=180:
                keyboard.press(Key.right) # the main point keay board decision function for right
                keyboard.release(Key.right)# the main point keay board decision function for left
            else:
                keyboard.press(Key.left)
                keyboard.release(Key.left)

        # Save our image
        # cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
            cv2.rectangle(image, (0, 0), (355, 73), (214, 44, 53)) #ractangle viwing window
            cv2.putText(image, 'Direction', (15, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)#image text
            cv2.putText(image, "Left" if angle >180 else "Right",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Tracking', image)# showing the image of tracking hand

        if cv2.waitKey(10) & 0xFF == ord('q'): #termininate key
            break


cap.release()
cv2.destroyAllWindows()
