import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
from pupil_apriltags import Detector

TAG_ID_TO_KEY = {15: 'tl', 126: 'tr', 3: 'bl', 47: 'br'}

# --- Utility Functions ---
def row_interp(row):
    row = row.astype(np.float32)
    mask = row == 0
    if np.all(mask):
        return row
    indices = np.arange(len(row))
    row[mask] = np.interp(indices[mask], indices[~mask], row[~mask])
    return row

def print_tag_dict():
    for k in ['tl', 'tr', 'bl', 'br']:
        if k in tag_dict:
            v = tag_dict[k]
            print(f"Key: {k}")
            print(f"  Tag ID: {v['tag_id']}")
            print(f"  Center: {v['center']}")

def get_tag_dict():
    return tag_dict

def store_tags(tag_dict, filename="april_tags.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(tag_dict, file)
    print(f"Tags stored in {filename}")
    print_tag_dict()

def rectify_color_with_tag_centers(color_image, tag_dict):
    if len(tag_dict) < 4:
        raise ValueError("Need all 4 corner tags: 'tl', 'tr', 'bl', 'br'")

    points = np.array([tag_dict[k]['center'] for k in ['tl', 'tr', 'bl', 'br']], dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()
    ordered_src = np.zeros((4, 2), dtype=np.float32)
    ordered_src[0] = points[np.argmin(s)]
    ordered_src[1] = points[np.argmin(diff)]
    ordered_src[2] = points[np.argmax(s)]
    ordered_src[3] = points[np.argmax(diff)]

    width = 505
    height = 378

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    return cv2.warpPerspective(color_image, M, (width, height), flags=cv2.INTER_CUBIC)

def rectify_depth_with_tag_centers(array, tag_dict):
    if len(tag_dict) < 4:
        raise ValueError("Need all 4 corner tags: 'tl', 'tr', 'bl', 'br'")

    points = np.array([tag_dict[k]['center'] for k in ['tl', 'tr', 'bl', 'br']], dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()
    ordered_src = np.zeros((4, 2), dtype=np.float32)
    ordered_src[0] = points[np.argmin(s)]
    ordered_src[1] = points[np.argmin(diff)]
    ordered_src[2] = points[np.argmax(s)]
    ordered_src[3] = points[np.argmax(diff)]

    width = 505
    height = 378

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    return cv2.warpPerspective(array, M, (width, height), flags=cv2.INTER_NEAREST)

def get_colormap_image(array, colormap=cv2.COLORMAP_TURBO):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    valid = array[array > 0]
    min_val = np.percentile(valid, 1)
    max_val = np.percentile(valid, 99)
    array = np.clip(array, min_val, max_val)
    norm = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)

# --- RealSense Init ---
ctx = rs.context()
if len(ctx.devices) == 0:
    raise RuntimeError("No RealSense device detected.")

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
pipe.start(cfg)
for _ in range(5):
    pipe.wait_for_frames()

spat_filter = rs.spatial_filter()
hole_filter = rs.hole_filling_filter()
hole_filter.set_option(rs.option.holes_fill, 2)
detector = Detector(families="tagStandard41h12")
tag_dict = {}

print("Searching for all 4 AprilTags...")

while True:
    frames = pipe.wait_for_frames()
    aligned = rs.align(rs.stream.color).process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    NUM_FRAMES = 5
    depth_stack = []
    for _ in range(NUM_FRAMES):
        avg_frames = rs.align(rs.stream.color).process(pipe.wait_for_frames())
        frame = avg_frames.get_depth_frame()
        filtered = spat_filter.process(frame)
        filtered = hole_filter.process(filtered)
        depth = np.asanyarray(filtered.get_data()).astype(np.float32)
        depth = np.apply_along_axis(row_interp, axis=1, arr=depth)
        depth_stack.append(depth)

    depth_image = np.mean(depth_stack, axis=0)
    print("Depth image min/max:", np.min(depth_image), np.max(depth_image))

    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite("checkalign.png", color_image)

    gray_np = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_np, estimate_tag_pose=False)
    tag_dict = {}
    for detection in detections:
        key = TAG_ID_TO_KEY.get(detection.tag_id)
        if key:
            tag_dict[key] = {
                'tag_id': detection.tag_id,
                'center': detection.center,
            }

    if all(k in tag_dict for k in ['tl', 'tr', 'bl', 'br']):
        print("Found all 4 AprilTags!")
        store_tags(tag_dict)
        break
    else:
        print(f"Tags found: {list(tag_dict.keys())} — still searching...")

def deproject_depth_point(x, y, depth):
    return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth))

def get_sandbox_transform(tag_dict, depth_image):
    world_points = []
    for key in ['tl', 'tr', 'bl', 'br']:
        x, y = tag_dict[key]['center']
        col = np.clip(int(round(x)), 0, depth_image.shape[1] - 1)
        row = np.clip(int(round(y)), 0, depth_image.shape[0] - 1)
        z = depth_image[row, col]
        if z == 0:
            raise ValueError(f"Invalid depth for tag {key} at ({row}, {col})")
        world_points.append(deproject_depth_point(col, row, z))
    origin = world_points[0]
    x_axis = world_points[1] - origin
    y_axis = world_points[3] - origin
    x_axis /= np.linalg.norm(x_axis)
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = -R.T @ origin
    print("Rotation matrix R:\n", R)
    print("Translation vector T:\n", T)
    return R.T, T

def generate_elevation_map(depth_image, R, T):
    h, w = depth_image.shape
    elevation_map = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            z = depth_image[row, col]
            if z == 0:
                elevation_map[row, col] = 0
                continue
            P_cam = deproject_depth_point(col, row, z)
            P_box = R @ P_cam + T
            elevation_map[row, col] = P_box[2]
    return elevation_map

# --- Compute Relative Depth ---
R, T = get_sandbox_transform(tag_dict, depth_image)
elevation_map = generate_elevation_map(depth_image, R, T)

tag_heights = []
for key in ['tl', 'tr', 'bl', 'br']:
    x, y = tag_dict[key]['center']
    col = np.clip(int(round(x)), 0, depth_image.shape[1] - 1)
    row = np.clip(int(round(y)), 0, depth_image.shape[0] - 1)
    z = depth_image[row, col]
    if z != 0:
        P_cam = deproject_depth_point(col, row, z)
        P_box = R @ P_cam + T
        tag_heights.append(P_box[2])
        
avg_tag_height = np.mean(tag_heights)
elevation_map = elevation_map - avg_tag_height
relative_elevation = 1000 - (elevation_map * 10 )
relative_elevation = rectify_depth_with_tag_centers(relative_elevation, tag_dict)

np.savez("calibration_data.npz", R=R, T=T, tag_dict=tag_dict)

colored_depth = get_colormap_image(relative_elevation)
cv2.imwrite("depthColorMap.png", colored_depth)
np.savetxt("depthdata.txt", relative_elevation, fmt='%d')

rectified_color = rectify_color_with_tag_centers(color_image, tag_dict)
cv2.imwrite("color_original.png", color_image)
cv2.imwrite("color_rectified.png", rectified_color)

print("Color image shape:", color_image.shape)
print("Depth image shape:", depth_image.shape)
print("Rectified color shape:", rectified_color.shape)
print("Colored elevation map shape:", colored_depth.shape)

pipe.stop()