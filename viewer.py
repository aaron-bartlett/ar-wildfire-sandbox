import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
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

def get_colormap_image(array, colormap=cv2.COLORMAP_TURBO):
    norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)
"""
def get_colormap_image(array, colormap=cv2.COLORMAP_TURBO):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional: clip out crazy values (change these as needed)
    array = np.clip(array, -20, 200)  # adjust min/max for best contrast

    # Normalize to 0–255
    norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Optional: Contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    return cv2.applyColorMap(norm, colormap)

"""
"""def fit_plane_to_tags(depth_image, tag_dict):
    Fit a plane to the AprilTag centers and return the fitted plane (as a 2D array).
    h, w = depth_image.shape
    points = []
    for key in ['tl', 'tr', 'bl', 'br']:
        x, y = tag_dict[key]['center']
        col = np.clip(int(round(x)), 0, w - 1)
        row = np.clip(int(round(y)), 0, h - 1)
        z = depth_image[row, col]
        points.append((col, row, z))

    points = np.array(points)
    X = points[:, :2]
    Z = points[:, 2]

    # Add bias term for constant
    A = np.c_[X, np.ones(X.shape[0])]
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    a, b, c = coeffs

    # Generate fitted plane for entire image
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    plane = a * xx + b * yy + c
    return plane
"""
def rectify_array_with_tag_centers(array, tag_dict):
    if len(tag_dict) < 4:
        raise ValueError("Need all 4 corner tags: 'tl', 'tr', 'bl', 'br'")
    points = np.array([tag_dict[k]['center'] for k in ['tl', 'tr', 'bl', 'br']], dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()
    ordered_src = np.zeros((4, 2), dtype=np.float32)
    ordered_src[0] = points[np.argmin(s)]  # Top-left
    ordered_src[2] = points[np.argmax(s)]  # Bottom-right
    ordered_src[1] = points[np.argmin(diff)]  # Top-right
    ordered_src[3] = points[np.argmax(diff)]  # Bottom-left

    width = int(np.linalg.norm(ordered_src[0] - ordered_src[1]))
    height = int(np.linalg.norm(ordered_src[0] - ordered_src[3]))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    rectified = cv2.warpPerspective(array, M, (width, height), flags=cv2.INTER_NEAREST)
    return rectified

def compute_tag_plane_depth(depth_array, tag_dict):
    centers = []
    for key in ['tl', 'tr', 'bl', 'br']:
        x, y = tag_dict[key]['center']
        col = np.clip(int(round(x)), 0, depth_array.shape[1] - 1)
        row = np.clip(int(round(y)), 0, depth_array.shape[0] - 1)
        centers.append(depth_array[row, col])
    return np.mean(centers)

def get_colormap_image(array, colormap=cv2.COLORMAP_TURBO):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extremes before visualizing
    array = np.clip(array, 50, 150)  # adjust as needed

    norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)
# --- RealSense Initialization ---
ctx = rs.context()
if len(ctx.devices) == 0:
    raise RuntimeError("No RealSense device detected.")

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipe.start(cfg)

# --- Frame Capture ---
frame = pipe.wait_for_frames()
aligned = rs.align(rs.stream.color).process(frame)
depth_frame = aligned.get_depth_frame()
color = aligned.get_color_frame()
depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

# --- OPTIONAL: Apply RealSense Filtering ---
"""dec_filter = rs.decimation_filter()
spat_filter = rs.spatial_filter()
temp_filter = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

filtered = dec_filter.process(depth_frame)
filtered = spat_filter.process(filtered)
"""
# ✅ Use the filtered depth data
depth_image = np.asanyarray(depth_frame.get_data())
depth_image = np.apply_along_axis(row_interp, axis=1, arr=depth_image)

# ✅ Use original color frame
color_image = np.asanyarray(color.get_data())


print("Depth image shape:", depth_image.shape)


# --- AprilTag Detection ---
cv2.imwrite("color_raw.png", color_image)
gray_img = Image.open("color_raw.png").convert("L")
gray_np = np.array(gray_img)

detector = Detector(families="tagStandard41h12")
detections = detector.detect(gray_np, estimate_tag_pose=False)

tag_dict = {}
for detection in detections:
    key = TAG_ID_TO_KEY.get(detection.tag_id)
    if key:
        tag_dict[key] = {
            'tag_id': detection.tag_id,
            'center': detection.center,
            # 'corners': detection.corners
        }

# --- Check Tags and Rectify Depth Image ---
def print_tag_dict():
    for k in ['tl', 'tr', 'bl', 'br']:
        if k in tag_dict:
            v = tag_dict[k]
            print(f"Key: {k}")
            print(f"  Tag ID: {v['tag_id']}")
            print(f"  Center: {v['center']}")
            # print(f"  Corners: {v['corners']}\n")

def get_tag_dict():
    return tag_dict

def deproject_depth_point(x, y, depth):
    return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth))

def get_sandbox_transform(tag_dict, depth_image):
    world_points = []
    for key in ['tl', 'tr', 'bl', 'br']:
        x, y = tag_dict[key]['center']
        col = np.clip(int(round(x)), 0, depth_image.shape[1] - 1)
        row = np.clip(int(round(y)), 0, depth_image.shape[0] - 1)

        z = depth_image[row, col]
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
# --- Compute Relative Elevation ---
R, T = get_sandbox_transform(tag_dict, depth_image)
relative_elevation = generate_elevation_map(depth_image, R, T)

# --- Rectify for visualization only ---
relative_elevation = rectify_array_with_tag_centers(relative_elevation, tag_dict)

# Shift elevation map to all-positive (zero = sandbox floor)
relative_elevation -= np.min(relative_elevation)

# Clip out crazy stuff (e.g., fingers, trash, spikes)
# Analyze actual elevation range
min_elev = np.percentile(relative_elevation, 1)
max_elev = np.percentile(relative_elevation, 99)
print(f"Elevation range (1%-99%): {min_elev:.2f} to {max_elev:.2f}")

# Clip only extreme outliers
relative_elevation = np.clip(relative_elevation, min_elev, max_elev)

# --- Colorize and Save ---
colored_depth = get_colormap_image(relative_elevation)
cv2.imwrite("depth_colormap.png", colored_depth)
np.savetxt("depthdata.txt", relative_elevation, fmt='%d')

pipe.stop()
