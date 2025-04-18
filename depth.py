import cv2
import numpy as np
import pygame
import pyrealsense2 as rs
import mediapipe as mp
import os
import time


HEIGHT = 378
WIDTH = 505
BASE_HEIGHT = 0.
MAX_HEIGHT = 100.
COLOR_CONSTANTS =  [(180, 160, 140), (180, 150, 100), (150, 150, 70), (130, 150, 70), (80, 130, 50), (40, 110, 70), (30, 100, 80), (30, 70, 100), (10, 20, 80)]


class Depth():
    def __init__(self, pygame_screen):

        print("Entered depth.py")
        self.pygame_screen = pygame_screen

        self.last_hand_state = False


        # grab tag_dict for perspective transforms
        data = np.load("calibration_data.npz", allow_pickle=True)
        self.R = data["R"]
        self.T = data["T"]
        self.tag_dict = data["tag_dict"].item()


        # realsense initialization
        context = rs.context()
        if len(context.devices) == 0:
            raise RuntimeError("No Realsense device detected.")

        self.pipe = rs.pipeline()
        cfg = rs.config()

        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

        self.pipe.start(cfg)

        self.align = rs.align(rs.stream.color)
        for _ in range(5):
            self.align.process(self.pipe.wait_for_frames())  # Skip startup noise

        self.spat_filter = rs.spatial_filter()
        self.hole_filter = rs.hole_filling_filter()
        self.hole_filter.set_option(rs.option.holes_fill, 2)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def get_height_surface(self, pygame_screen):

        array = np.load("./data/depth_camera_input.npy")
        height, width = array.shape
        array = np.clip(array, BASE_HEIGHT, MAX_HEIGHT)
        bins = np.linspace(BASE_HEIGHT, MAX_HEIGHT, num=9)
        categories = np.digitize(array, bins) - 1 

        rgb_array = np.array([COLOR_CONSTANTS[c] for c in categories.flatten()])
        rgb_array = rgb_array.reshape(*array.shape, 3).astype(np.uint8)

        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        height_surface =  pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
        height_surface = pygame.transform.scale(height_surface, (screen_w, screen_h))
            
        pygame_screen.blit(height_surface, (0, 0))
        pygame.display.update()
        '''
        clock = pygame.time.Clock()
        start = time.time()
        
        while time.time() - start < 5:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            clock.tick(60)
        '''
        return



    # retification function for perspective transforms
    def rectify_color_with_tag_centers(self, color_image, tag_dict):
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

    # -----mediapipe hand detection model initialization----------


    def hands_in_frame(self):
        return self.hand_detected

    # --- Utility Functions ---
    def row_interp(row):
        row = row.astype(np.float32)
        mask = row == 0
        if np.all(mask):
            return row
        indices = np.arange(len(row))
        row[mask] = np.interp(indices[mask], indices[~mask], row[~mask])
        return row

    def deproject_depth_point(self, intrin, x, y, depth):
        return np.array(rs.rs2_deproject_pixel_to_point(intrin, [x, y], depth))

    def generate_elevation_map(self, depth_image, R, T, intrin):
        h, w = depth_image.shape
        elevation_map = np.zeros((h, w), dtype=np.float32)
        for row in range(h):
            for col in range(w):
                z = depth_image[row, col]
                if z == 0:
                    elevation_map[row, col] = 0
                    continue
                P_cam = self.deproject_depth_point(intrin, col, row, z)
                P_box = R @ P_cam + T
                elevation_map[row, col] = P_box[2]
        return elevation_map

    def rectify_depth_with_tag_centers(self, array, tag_dict):
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

    def get_colormap_image(self, array, colormap=cv2.COLORMAP_TURBO):
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        valid = array[array > 0]
        min_val = np.percentile(valid, 1)
        max_val = np.percentile(valid, 99)
        array = np.clip(array, min_val, max_val)
        norm = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, colormap)

    # --- Main Capture ---
    def grab_depth_map(self):
    
        frames = self.align.process(self.pipe.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        # Filter and interpolate
        filtered = self.spat_filter.process(depth_frame)
        filtered = self.hole_filter.process(filtered)
        depth_image = np.asanyarray(filtered.get_data()).astype(np.float32)
        depth_image = np.apply_along_axis(self.row_interp, axis=1, arr=depth_image)

        # Step 1: generate elevation map
        elevation_map = self.generate_elevation_map(depth_image, self.R, self.T, intrin)


        # Step 2: compute average AprilTag height
        tag_heights = []
        for key in ['tl', 'tr', 'bl', 'br']:
            x, y = self.tag_dict[key]['center']
            col = int(round(x))
            row = int(round(y))
            z = depth_image[row, col]
            if z != 0:
                P_cam = self.deproject_depth_point(intrin, col, row, z)
                P_box = self.R @ P_cam + self.T
                tag_heights.append(P_box[2])

        avg_tag_height = np.mean(tag_heights)

        # Step 3: normalize using your formula
        elevation_map = elevation_map - avg_tag_height
        relative_elevation = 1000 - (elevation_map * 10)

        # Step 4: rectify and clip
        relative_elevation = self.rectify_depth_with_tag_centers(relative_elevation, self.tag_dict)
        relative_elevation = np.clip(relative_elevation, 0, 2000)

        # Step 5: colorize and save
        colored = self.get_colormap_image(relative_elevation)
        cv2.imwrite("depth_colormap.png", colored)
        np.save("data/depth_camera_input.npy", relative_elevation)

        print("Depth capture and elevation mapping complete.")
        print("Colored elevation map shape:", relative_elevation.shape)  # H x W x 3

        return relative_elevation

    def grab_hand_position(self):
        global last_hand_state

        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while True:
                frame = self.pipe.wait_for_frames()
                aligned = rs.align(rs.stream.color).process(frame)
                color_frame = aligned.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                rect_image = self.rectify_color_with_tag_centers(color_image, self.tag_dict)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                rect_image.flags.writeable = False
                rect_image = cv2.cvtColor(rect_image, cv2.COLOR_BGR2RGB)
                results = hands.process(rect_image)
                self.hand_detected = results.multi_hand_landmarks is not None

                #global screen, screen_h, screen_w
                
                '''
                pygame.init()
                os.environ['SDL_VIDEO_CENTERED'] = "1"
                info = pygame.display.Info()
                screen_w, screen_h = info.current_w, info.current_h
                screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
                
                pygame.display.set_caption("Height Surface Viewer")
                '''
                
                if self.hand_detected and not last_hand_state:
                    print("[INFO] Hands entered frame")
                    self.get_height_surface()
                elif not self.hand_detected and last_hand_state:
                    print("[INFO] Hands removed from frame")
                    time.sleep(5)
                    self.grab_depth_map()
                    self.get_height_surface()

                last_hand_state = self.hand_detected  # update for next loop

                # Draw the hand annotations on the image.
                rect_image.flags.writeable = True
                rect_image = cv2.cvtColor(rect_image, cv2.COLOR_RGB2BGR)
                if self.hand_detected:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            rect_image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', rect_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.pipe.stop()
                    #pygame.quit()
                    print("Hand tracking is complete")
                    break


def main():
    return
    #grab_hand_position()
        
        
if __name__ == "__main__":
    main()
