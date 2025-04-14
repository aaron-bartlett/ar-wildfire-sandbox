import pyrealsense2 as rs
import cv2
import numpy as np
from calibration import get_tag_dict, print_tag_dict

def extract_tag(tag_dict):
    tl = tag_dict['tl']['center']
    tr = tag_dict['tr']['center']
    bl = tag_dict['bl']['center']
    br = tag_dict['br']['center']

    src_pts = [tl, tr, br, bl]
    return src_pts

scale = 10
width, height = 40, 30
warped_size = (int(width * scale), int(height * scale))

src_pts = np.array(extract_tag(get_tag_dict()), dtype=np.float32)
print_tag_dict()

dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32) * scale

M, _ = cv2.findHomography(src_pts, dst_pts)

print("H\n\n", M)

warped_image = cv2.warpPerspective(color_image, M, warped_size)

orig_resized = cv2.resize(color_image, (warped_image.shape[1], warped_image.shape[0]))

side_by_side = np.hstack((orig_resized, warped_image))
cv2.imwrite("original.png", orig_resized)
cv2.imwrite("new.png", warped_image)
# cv2.imshow("Original (left) | Warped Top-Down (right)", side_by_side)

# cv2.destroyAllWindows()
pipe.stop()