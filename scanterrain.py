import pyrealsense2 as rs
import numpy as np
import cv2

# --- Load Calibration ---
data = np.load("calibration_data.npz", allow_pickle=True)
R = data["R"]
T = data["T"]
tag_dict = data["tag_dict"].item()

# --- RealSense Init ---
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipe.start(cfg)

align = rs.align(rs.stream.color)
for _ in range(5):
    align.process(pipe.wait_for_frames())  # Skip startup noise

spat_filter = rs.spatial_filter()
hole_filter = rs.hole_filling_filter()
hole_filter.set_option(rs.option.holes_fill, 2)

# --- Utility Functions ---
def row_interp(row):
    row = row.astype(np.float32)
    mask = row == 0
    if np.all(mask):
        return row
    indices = np.arange(len(row))
    row[mask] = np.interp(indices[mask], indices[~mask], row[~mask])
    return row

def deproject_depth_point(intrin, x, y, depth):
    return np.array(rs.rs2_deproject_pixel_to_point(intrin, [x, y], depth))

def generate_elevation_map(depth_image, R, T, intrin):
    h, w = depth_image.shape
    elevation_map = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            z = depth_image[row, col]
            if z == 0:
                elevation_map[row, col] = 0
                continue
            P_cam = deproject_depth_point(intrin, col, row, z)
            P_box = R @ P_cam + T
            elevation_map[row, col] = P_box[2]
    return elevation_map

def rectify_depth_with_tag_centers(array, tag_dict):
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

    # Fixed output resolution
    width = 505
    height = 378

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered_src, dst_pts)
    rectified = cv2.warpPerspective(array, M, (width, height), flags=cv2.INTER_NEAREST)
    return rectified

def get_colormap_image(array, colormap=cv2.COLORMAP_TURBO):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    valid = array[array > 0]
    min_val = np.percentile(valid, 1)
    max_val = np.percentile(valid, 99)
    array = np.clip(array, min_val, max_val)
    norm = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, colormap)

# --- Main Capture ---
frames = align.process(pipe.wait_for_frames())
depth_frame = frames.get_depth_frame()
intrin = depth_frame.profile.as_video_stream_profile().intrinsics

# Filter and interpolate
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

# Step 1: generate elevation map
elevation_map = generate_elevation_map(depth_image, R, T, intrin)

# Step 2: compute average AprilTag height
tag_heights = []
for key in ['tl', 'tr', 'bl', 'br']:
    x, y = tag_dict[key]['center']
    col = int(round(x))
    row = int(round(y))
    z = depth_image[row, col]
    if z != 0:
        P_cam = deproject_depth_point(intrin, col, row, z)
        P_box = R @ P_cam + T
        tag_heights.append(P_box[2])

avg_tag_height = np.mean(tag_heights)

# Step 3: normalize using your formula
elevation_map = elevation_map - avg_tag_height
relative_elevation = 1000 - (elevation_map * 10)

# Step 4: rectify and clip
relative_elevation = rectify_depth_with_tag_centers(relative_elevation, tag_dict)
relative_elevation = np.clip(relative_elevation, 0, 2000)

# Step 5: save output
colored = get_colormap_image(relative_elevation)
cv2.imwrite("depth_colormap.png", colored)
np.savetxt("depthdata.txt", relative_elevation, fmt='%d')

pipe.stop()
print("Depth capture and elevation mapping complete.")
print("Colored elevation map shape:", relative_elevation.shape)