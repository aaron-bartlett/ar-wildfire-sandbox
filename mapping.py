import pyrealsense2 as rs
import numpy as np
import cv2

# --- Load Calibration ---
data = np.load("calibration_data.npz", allow_pickle=True)
R = data["R"]
T = data["T"]
tag_centers_array = np.array([data[k]['center'] for k in ['tl', 'tr', 'bl', 'br'] if k in data])  # Dictionary of tag center coordinates

# --- RealSense Init ---
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
pipe.start(cfg)

for _ in range(5):
    pipe.wait_for_frames()  # Skip startup noise

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

def get_colormap_image(array, colormap=cv2.COLORMAP_TURBO):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    min_elev = np.percentile(array, 1)
    max_elev = np.percentile(array, 99)
    array = np.clip(array, min_elev, max_elev)
    array = ((array - min_elev) / (max_elev - min_elev) * 255).astype(np.uint8)
    return cv2.applyColorMap(array, colormap)

# --- Main Capture ---
frames = pipe.wait_for_frames()
depth_frame = frames.get_depth_frame()
intrin = depth_frame.profile.as_video_stream_profile().intrinsics

# Filter and interpolate
filtered = spat_filter.process(depth_frame)
filtered = hole_filter.process(filtered)
depth_image = np.asanyarray(filtered.get_data()).astype(np.float32)
depth_image = np.apply_along_axis(row_interp, axis=1, arr=depth_image)

# Generate elevation map using saved transform
elevation_map = generate_elevation_map(depth_image, R, T, intrin)
elevation_map -= np.min(elevation_map)

# Save result
colorized = get_colormap_image(elevation_map)
cv2.imwrite("depth_colormap.png", colorized)
np.savetxt("depthdata.txt", elevation_map, fmt='%d')

pipe.stop()
print("Depth capture and elevation mapping complete.")
