import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

print("Entered object.py")

# grab tag_dict for perspective transforms
data = np.load("calibration_data.npz", allow_pickle=True)
R = data["R"]
T = data["T"]
tag_dict = data["tag_dict"].item()


# retification function for perspective transforms
def rectify_color_with_tag_centers(color_image, tag_dict):
    if len(tag_dict) < 4:
        raise ValueError("Need all 4 corner tags: 'tl', 'tr', 'bl', 'br'")
    
    points = np.array([tag_dict[k]['center'] for k in ['tl', 'tr', 'bl', 'br']], dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()
    ordered_src = np.zeros((4, 2), dtype=np.float32)
    ordered_src[0] = points[np.argmin(s)]      # Top-left
    ordered_src[1] = points[np.argmin(diff)]   # Top-right
    ordered_src[2] = points[np.argmax(s)]      # Bottom-right
    ordered_src[3] = points[np.argmax(diff)]   # Bottom-left

    # Fixed output size
    width = 505
    height = 378

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    rectified_color = cv2.warpPerspective(color_image, M, (width, height), flags=cv2.INTER_CUBIC)
    return rectified_color

print("Starting camera")

context = rs.context()
if len(context.devices) == 0:
    raise RuntimeError("No Realsense device detected.")

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipe.start(cfg)

for _ in range(5):
    pipe.wait_for_frames()

print("Importing model")

# train6 is latest model running
model = YOLO("runs/detect/train6/weights/best.pt")


# Run YOLO object detection.
def detect_objects(frame):
    results = model(frame, conf=0.5)
    return results

def main():
    print("In main")
    while True:
        frame = pipe.wait_for_frames()
        aligned = rs.align(rs.stream.color).process(frame)
        
        print("Grabbing colors...")

        color_frame = aligned.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        rect_image = rectify_color_with_tag_centers(color_image, tag_dict)
    
        # object detection
        print("Detecting objects...")
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

                #text = f"{label} {x1} {x2} {y1} {y2} {conf:.2f}"
                text = f"{label} {conf:.2f}"
                cv2.putText(rect_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Object Detection', rect_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saving data to final_objects.txt...")
            with open("data/final_objects.txt", "w") as f:
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    boxes_cls = result.boxes.cls.cpu().numpy()
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cls = int(boxes_cls[i])
                        label = model.names[cls]
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        center_y = int( float(center_y - 50) * 378 / 328)
                        f.write(f"{label},{center_x},{center_y}\n")
            break

    pipe.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()