import cv2
import pyrealsense2 as rs
import mediapipe as mp
from calibration import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

context = rs.context()
if len(context.devices) == 0:
    raise RuntimeError("No Realsense device detected.")

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipe.start(cfg)

for _ in range(5):
    pipe.wait_for_frames()


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while True:
    frame = pipe.wait_for_frames()
    aligned = rs.align(rs.stream.color).process(frame)
    color_frame = aligned.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    rect_image = rectify_color_with_tag_centers(color_image, tag_dict)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    rect_image.flags.writeable = False
    rect_image = cv2.cvtColor(rect_image, cv2.COLOR_BGR2RGB)
    results = hands.process(rect_image)

    # Draw the hand annotations on the image.
    rect_image.flags.writeable = True
    rect_image = cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      hand_detected = True
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            rect_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    else:
       hand_detected = False
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(rect_image, 1))
    print("hand_detected: ", hand_detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()