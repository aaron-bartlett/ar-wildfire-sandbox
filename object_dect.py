import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import pickle
from ultralytics import YOLO
from calibration import *

# step 1: nab april tags location
def load_tags(filename="april_tags.pkl"):
    try:
        with open(filename, "rb") as file:
            tag_dict = pickle.load(file)
        print(f"Tags loaded from {filename}")
        return tag_dict
    except FileNotFoundError:
        print(f"{filename} not found. Returning an empty dictionary.")
        return {}


tag_dict = load_tags()
print(tag_dict)

context = rs.context()
if len(context.devices) == 0:
    raise RuntimeError("No Realsense device detected.")

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe.start(cfg)

for _ in range(5):
    pipe.wait_for_frames()

model = YOLO("runs/detect/train7/weights/best.pt")

# Run YOLO object detection.
def detect_objects(frame):
    results = model(frame, conf=0.5)
    return results

def main():
    while True:
        frame = pipe.wait_for_frames()
        aligned = rs.align(rs.stream.color).process(frame)

        color_frame = aligned.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        rect_image = rectify_color_with_tag_centers(color_image, get_tag_dict())

        # object detection
        results = detect_objects(rect_image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            boxes_cls = result.boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = boxes_conf[i]
                cls = int(boxes_cls[i])
                label = model.names[cls]
                cv2.rectangle(rect_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                ### think about adding thresholds for the bounding boxes to reduce the number of false positives

                text = f"{label} {conf:.2f}"
                cv2.putText(rect_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # hand detection



        cv2.imshow('Object Detection', rect_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()