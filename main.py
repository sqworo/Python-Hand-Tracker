import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

ball_pos = np.array([300, 300])
ball_velocity = np.array([5, 5])
ball_radius = 20

def detect_collision_with_line(ball_pos, p1, p2, radius):
    px, py = ball_pos
    x1, y1 = p1
    x2, y2 = p2

    line_mag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if line_mag < 0.000001:
        return False

    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    closest_point = (x1 + u * (x2 - x1), y1 + u * (y2 - y1))

    dist = math.sqrt((closest_point[0] - px) ** 2 + (closest_point[1] - py) ** 2)
    
    if dist <= radius:
        return True
    return False

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    black_img = np.zeros_like(img)

    index_finger_1 = None
    index_finger_2 = None

    # Si se detectan manos
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(black_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = black_img.shape
            index_finger_x = int(hand_landmarks.landmark[8].x * w)
            index_finger_y = int(hand_landmarks.landmark[8].y * h)

            if idx == 0:
                index_finger_1 = (index_finger_x, index_finger_y)
            elif idx == 1:
                index_finger_2 = (index_finger_x, index_finger_y)

    if index_finger_1 and index_finger_2:
        cv2.line(black_img, index_finger_1, index_finger_2, (0, 0, 255), 5)
        if detect_collision_with_line(ball_pos, index_finger_1, index_finger_2, ball_radius):
            ball_velocity = -ball_velocity

    ball_pos += ball_velocity

    if ball_pos[0] - ball_radius <= 0 or ball_pos[0] + ball_radius >= black_img.shape[1]:
        ball_velocity[0] = -ball_velocity[0]
    if ball_pos[1] - ball_radius <= 0 or ball_pos[1] + ball_radius >= black_img.shape[0]:
        ball_velocity[1] = -ball_velocity[1]

    cv2.circle(black_img, (ball_pos[0], ball_pos[1]), ball_radius, (0, 255, 0), -1)

    cv2.imshow("Hand Tracker", black_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
