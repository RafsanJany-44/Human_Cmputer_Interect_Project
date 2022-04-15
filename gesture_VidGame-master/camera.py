
import mediapipe as md
import cv2
import numpy as npe
import uuid
import os
from pynput.keyboard import Key, Controller

md_graphics = md.solutions.drawing_utils # taking drwaing utils
md_finger = md.solutions.hands
mark_list =[[8,5,0]]  # ladbmarks of 8,5,0

def finger_visualization(graphics, results, mark_list):   # this line is drwaing the finger line and the text in the monitor and calculating degree
    # Loop through hands 
    for hand in results.multi_hand_landmarks:  #building p loop for hand marks
        # Loop through joint sets
        for j in mark_list: #travversing each segment of land marks
            p = npe.array([hand.landmark[j[0]].x, hand.landmark[j[0]].y])  # First coord
            q = npe.array([hand.landmark[j[1]].x, hand.landmark[j[1]].y])  # Second coord
            r = npe.array([hand.landmark[j[2]].x, hand.landmark[j[2]].y])  # Third coord

            rad = npe.arctan2(r[1] - q[1], r[0] - q[0]) - npe.arctan2(p[1] - q[1], p[0] - q[0])
            degree = npe.abs(rad * 180.0 / npe.pi) #calculating degree for decision left or right


            cv2.putText(graphics, str(round(degree, 2)), tuple(npe.multiply(q, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA) #puting text into the monitor
    return graphics, degree

def Labels(i, h, r):  #getting the level
    output = None
    for idx, classification in enumerate(r.multi_handedness):
        if classification.classification[0].i == i:
            # Process results
            l = classification.classification[0].l
            s = classification.classification[0].s
            t = '{} {}'.format(l, round(s, 2))

            # Extract Coordinates
            cds = tuple(npe.multiply(
                npe.array((h.landmark[md_finger.HandLandmark.WRIST].x, h.landmark[md_finger.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))

            result = t, cds

    return result


caption = cv2.VideoCapture(0) #for vedio capturing

with md_finger.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while caption.isOpened():
        ret, f = caption.read() #f = frame

        # BGR 2 RGB
        graphics = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        graphics = cv2.flip(graphics, 1)

        # Set flag
        graphics.flags.writeable = False

        # Detections
        results = hands.process(graphics)

        # Set flag to true
        graphics.flags.writeable = True

        # RGB 2 BGR
        graphics = cv2.cvtColor(graphics, cv2.COLOR_RGB2BGR)

        # Detections
        print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                md_graphics.draw_landmarks(graphics, hand, md_finger.HAND_CONNECTIONS,
                                    md_graphics.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    md_graphics.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                    ) # drwaing each land mark and lines in hand

                # Render left or right detection
                if Labels(num, hand, results):
                    text, coord = Labels(num, hand, results)
                    cv2.putText(graphics, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw angles to graphics from joint list
            graphics, degree = finger_visualization(graphics, results, mark_list) #calculating angel and drwain in graphics
            keyboard = Controller() #calling keayboard controler
            if degree<=180:
                keyboard.press(Key.right) # the main point keay board decision function for right
                keyboard.release(Key.right)# the main point keay board decision function for left
            else:
                keyboard.press(Key.left)
                keyboard.release(Key.left)

        # Save our graphics
        # cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), graphics)
            cv2.rectangle(graphics, (0, 0), (355, 73), (214, 44, 53)) #ractangle viwing window
            cv2.putText(graphics, 'Direction', (15, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)#graphics text
            cv2.putText(graphics, "Left" if degree >180 else "Right",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Tracking', graphics)# showing the graphics of tracking hand

        if cv2.waitKey(10) & 0xFF == ord('q'): #termininate key
            break


caption.release()
cv2.destroyAllWindows()
