import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pTime = 0
smooth_positions = {}

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, img = cap.read()

        if not success:
            print("Camera not detected")
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        hand_info = []

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):
                lmList = []

                for id, lm in enumerate(hand_landmarks.landmark):
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Handedness
                if results.multi_handedness:
                    hand_label = results.multi_handedness[
                        hand_index
                    ].classification[0].label
                else:
                    hand_label = f"Hand {hand_index + 1}"

                # Finger detection
                finger_tips = [8, 12, 16, 20]
                fingers = []

                for tip in finger_tips:
                    if lmList[tip][2] < lmList[tip - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Thumb detection
                if hand_label == "Right":
                    thumb_open = lmList[4][1] < lmList[3][1]
                else:
                    thumb_open = lmList[4][1] > lmList[3][1]

                fingers.append(1 if thumb_open else 0)

                total_fingers = sum(fingers)

                # Gesture recognition
                if fingers == [1, 1, 1, 1, 1]:
                    gesture = "Open Palm"
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture = "Fist"
                elif fingers == [1, 1, 0, 0, 0]:
                    gesture = "Peace"
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "One Finger"
                elif fingers == [0, 0, 0, 0, 1]:
                    gesture = "Thumbs Up"
                else:
                    gesture = "Unknown"

                # Wrist position
                cx, cy = lmList[0][1], lmList[0][2]

                # Independent smoothing for each hand
                if hand_label not in smooth_positions:
                    smooth_positions[hand_label] = (cx, cy)

                old_x, old_y = smooth_positions[hand_label]

                smooth_x = int(old_x * 0.7 + cx * 0.3)
                smooth_y = int(old_y * 0.7 + cy * 0.3)

                smooth_positions[hand_label] = (smooth_x, smooth_y)

                # Draw landmarks
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Draw wrist marker
                cv2.circle(
                    img,
                    (smooth_x, smooth_y),
                    10,
                    (255, 0, 0),
                    cv2.FILLED
                )

                # Position coordinates
                cv2.putText(
                    img,
                    f"X:{smooth_x} Y:{smooth_y}",
                    (smooth_x + 15, smooth_y - 15),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 0),
                    2
                )

                hand_info.append(
                    (hand_label, total_fingers, gesture)
                )

        # FPS display
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 255),
            2
        )

        # Display information for each hand
        y_position = 80

        for label, count, gesture in hand_info:
            cv2.putText(
                img,
                f"{label}: {count} fingers - {gesture}",
                (10, y_position),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                (0, 255, 0),
                2
            )
            y_position += 35

        cv2.imshow("Hand Tracking System", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()