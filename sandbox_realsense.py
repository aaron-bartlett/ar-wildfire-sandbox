import pygame
import numpy as np
import threading
from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus
import time
from scipy.ndimage import zoom
import cv2
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

HEIGHT = 720
WIDTH = 1280
BASE_HEIGHT = 0.
MAX_HEIGHT = 2500.
#Color Constants from Red to Blue
#COLOR_CONSTANTS = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0), (255, 150, 0), (255, 0, 0)]
# Earth Tones
COLOR_CONSTANTS =  [(180, 160, 140), (180, 150, 100), (150, 150, 70), (130, 150, 70), (80, 130, 50), (40, 110, 70), (30, 100, 80), (30, 70, 100), (10, 20, 80)]

def interp_row(row):
     x = np.arange(len(row))
     mask = row != 0
     if np.sum(mask) >= 2:
         return np.interp(x, x[mask], row[mask])
     elif np.sum(mask) == 1:
         return np.full_like(row, row[mask][0], dtype=float)
     else:
         return row

def load_height_array_from_file(): # if loading from a txt file from a depth cam
    path="EECS498\ar-wildfire-sandbox\depthdata.txt"
    target_shape=(1080, 1440)
    try:
        array = np.loadtxt(path)
        zoom_factors = (
            target_shape[0] / array.shape[0],
            target_shape[1] / array.shape[1]
        )
        resized_array = zoom(array, zoom_factors, order=3)  

        return resized_array

    except Exception as e:
        print(f"Failed to load or resize height array from {path}: {e}")
        return np.zeros(target_shape, dtype=float)



def get_height_surface(array):

    height, width = array.shape
    array = np.clip(array, BASE_HEIGHT, MAX_HEIGHT)
    bins = np.linspace(BASE_HEIGHT, MAX_HEIGHT, num=9)
    categories = np.digitize(array, bins) - 1 

    rgb_array = np.array([COLOR_CONSTANTS[c] for c in categories.flatten()])
    rgb_array = rgb_array.reshape(*array.shape, 3).astype(np.uint8)
    #surface = pygame.Surface((width, height)) 
    #pygame.surfarray.blit_array(surface, np.transpose(rgb_array, (1, 0, 2)))
    return pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))


def make_line_surface(array): # turns all binary 1's too black
    height, width = array.shape
    
    # NOTE: Keep as width x height to match pygame
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[array == 1] = [255, 0, 0]  
    rgb_array[array < 1] = [255, 255, 255] 
    
    return pygame.surfarray.make_surface(rgb_array)

def read_line(mitigation_type): # take an array turn it into 1's and 0's 

    if mitigation_type not in [BurnStatus.FIRELINE, BurnStatus.SCRATCHLINE, BurnStatus.WETLINE]:
        print("Invalid mitigation type")
        return None
    
    # HAVE A CNN OR SOMETHING TO READ THE REALSENSE CAMERA INTO A 0/1 ARRAY FOR THE LINE
    # line_input = read_and_display_input()
    # TEST ARRAY
    if mitigation_type == BurnStatus.FIRELINE:
        line_input = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        line_input[:, (WIDTH // 2):(WIDTH // 2 + 3)] = 1
    else:
        line_input = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
        line_input[(HEIGHT // 2):(HEIGHT // 2 + 3), :] = 1
    return  np.transpose(line_input)


def listen_for_playpause():
    global user_input
    while user_input is not None:
        user_input_str = input("Press any key to set user_input to True, or 'N' to stop:\n")
        if user_input_str.strip().upper() == 'N':
            user_input = False
            print("User input set to False. Exiting input listener.")
        else:
            user_input = True
            print("User input received! Flag set to True.")

def detect_objects(frame):
    """Run YOLO object detection."""
    results = model(frame)
    return results

def real_realsense():
    while True:
        frame = pipe.wait_for_frames()

        depth = frame.get_depth_frame()
        color = frame.get_color_frame()

        depth_image = np.asanyarray(depth.get_data()) 
        color_image = np.asanyarray(color.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # Object Detection
        results = detect_objects(color_image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # # Line Detection
        # lines = detect_lines(gray_image)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)


        cv2.imshow("Depth Detection", depth_colormap)
        cv2.imshow('Object Detection', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()
    return depth_image

def modify_terrain(depth_array=None):

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Height Surface Viewer")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    running = False
                    
        frame = pipe.wait_for_frames()
        depth = frame.get_depth_frame()
        color = frame.get_color_frame()

        depth_array = np.asanyarray(depth.get_data()) 
        depth_array = np.apply_along_axis(interp_row, axis=1, arr=depth_array)
        #color_image = np.asanyarray(color.get_data())

        height_surface = get_height_surface(depth_array)
        screen.blit(height_surface, (0, 0))
        pygame.display.update()
    pygame.quit()
    return depth_array

def create_sim():
    config = Config("configs/camera_config.yml")

    sim = FireSimulation(config)

    sim.rendering = True

    return sim


def run_simulation():

    #global user_input
    #user_input = False

    sim = create_sim()
    #input_thread = threading.Thread(target=listen_for_playpause, daemon=True)
    #input_thread.start()

    for i in range (10):
        sim.run("5m")
        print(f"Sim Step {i}")
        input_flag = input("1: PLAY SIM, 2: ADD MITIGATION, 3: DISPLAY CONTOURS, 4 EXIT\n")
        if input_flag == "1":
            sim.run('1h')
        elif input_flag == "2":
            mitigation_input = input("Enter Mitigation Type (fireline, scratchline, wetline): ") 
        
            if(mitigation_input == "fireline"):
                mitigation_type = BurnStatus.FIRELINE
            elif(mitigation_input == "scratchline"):
                mitigation_type = BurnStatus.SCRATCHLINE
            elif(mitigation_input == "wetline"):
                mitigation_type = BurnStatus.WETLINE
            else:
                print("Invalid mitigation type")
                continue

            # GET HxW INPUT ARRAY W 1 FOR LINE 0 FOR NO INPUT
            line_input = read_line(mitigation_type)
            line_input_points  = np.argwhere(line_input == 1) # [[row, col]]
            line_input_points = [tuple(pair) for pair in line_input_points] # [(row, col)]
            mitigations = [(*coord, mitigation_type) for coord in line_input_points]

            mitigation_surface = make_line_surface(line_input)

            sim._blit_surface(mitigation_surface)
            print(mitigations[-1])
            time.sleep(2)
            sim.update_mitigation(mitigations)

        elif input_flag == "3":
            #row_values = np.linspace(0, 100, WIDTH)
            #height_array = np.tile(row_values, (HEIGHT, 1))
            y_coords = np.arange(HEIGHT-1, -1, -1).reshape(-1, 1)  # Height
            x_coords = np.arange(WIDTH)  # Width
            height_array = x_coords + y_coords
            height_surface = get_height_surface(height_array)
            sim._blit_surface(height_surface)

        else:
            return sim

        time.sleep(3)
        ''' TEMPORARILY UNUSED
        if user_input: # UPDATE WITH A WAY TO PAUSE/PLAY
            if True: # UPDATE WITH A WAY TO CHOOSE TO ADD A MITIGATION
                mitigation_input = input("Enter Mitigation Type (fireline, scratchline, wetline): ") 
                if(mitigation_input == "fireline"):
                    mitigation_type = BurnStatus.FIRELINE
                elif(mitigation_input == "scratchline"):
                    mitigation_type = BurnStatus.SCRATCHLINE
                elif(mitigation_input == "wetline"):
                    mitigation_type = BurnStatus.WETLINE
                else:
                    print("Invalid mitigation type")
                    continue
                mitigations = read_line(mitigation_type)
                
            sim.update_mitigation(mitigations)   
            user_input = False'
        '''

    #user_input = None
    return sim

context = rs.context()
if len(context.devices) == 0:
    raise RuntimeError("No Realsense device detected.")

pipe = rs.pipeline() 
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipe.start(cfg)

model = YOLO("runs/detect/train3/weights/best.pt")#YOLO('yolov8n.pt') # runs/detect/train/weights/best.pt

#depth_array = real_realsense()

#initial_terrain = modify_terrain(depth_array)
depth_array = modify_terrain()

np.save("./data/depth_camera_input.npy", depth_array)

sim = run_simulation()
sim.save_gif()