import numpy as np
from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus
import time
from scipy.ndimage import zoom
import cv2
#import pyrealsense2 as rs
#import torch
#from ultralytics import YOLO
import os
import pygame

HEIGHT = 378
WIDTH = 505
BASE_HEIGHT = 0.
MAX_HEIGHT = 100.
#Color Constants from Red to Blue
#COLOR_CONSTANTS = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0), (255, 150, 0), (255, 0, 0)]
# Earth Tones
COLOR_CONSTANTS =  [(180, 160, 140), (180, 150, 100), (150, 150, 70), (130, 150, 70), (80, 130, 50), (40, 110, 70), (30, 100, 80), (30, 70, 100), (10, 20, 80)]

running = False
calibrated = False
terrain_scanned = False
objects_scanned = False

def run_sim_loop(sim):
    global running
    while running:
        sim.run('5m')

    return

def initialize():
    if input("Calibrate? (y/n): ").lower() == 'y':
        # Calibrate system
        # get depthmap.txt
        os.system("python3 calibration.py")
        # display inital map contours
        # projector turns on
        get_height_surface()
        global calibrated
        calibrated = True
        return scan_options()
    else:
        return initialize()

def scan_terrain():
    print('scan_terrain')
    global terrain_scanned
    terrain_scanned = True
    # take the depth in
    os.system("python3 depth.py")
    
    print("ran depth.py")
    return

def scan_objects():
    print('scan_objects')
    global objects_scanned
    objects_scanned = True

    # project black
    get_black_surface()
    # run object detection
    os.system("python3 object.py")
    # project depth map again
    get_height_surface()

    return

def scan_options():
    global calibrated, terrain_scanned, objects_scanned
    selection = input(f"Scan Terrain ({terrain_scanned}), Scan Objects ({objects_scanned}), Re-Calibrate ({calibrated}), or Start Simulation? (t/o/c/start): ").lower()
    if selection == 't':
        scan_terrain()
        return scan_options()
    elif selection == 'o':  
        scan_objects()
        return scan_options()
    elif selection == 'c':
        return initialize()
    elif (selection == 's') | (selection == 'start'):
        if not calibrated:
            print("Please calibrate first.")
            return scan_options()
        if not terrain_scanned:
            print("Please scan terrain first.")
            return scan_options()
        if not objects_scanned:
            print("Please scan objects first.")
            return scan_options()
        else:
            print("Starting simulation...")
            return start_sim()
    else:
        print("Invalid selection. Please try again.")
        return scan_options()
    

def get_height_surface():

    array = np.load("./data/depth_camera_input.npy")
    height, width = array.shape
    array = np.clip(array, BASE_HEIGHT, MAX_HEIGHT)
    bins = np.linspace(BASE_HEIGHT, MAX_HEIGHT, num=9)
    categories = np.digitize(array, bins) - 1 

    rgb_array = np.array([COLOR_CONSTANTS[c] for c in categories.flatten()])
    rgb_array = rgb_array.reshape(*array.shape, 3).astype(np.uint8)
    #surface = pygame.Surface((width, height)) 
    #pygame.surfarray.blit_array(surface, np.transpose(rgb_array, (1, 0, 2)))

    pygame.init()
    height_surface =  pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Height Surface Viewer")
    screen.blit(height_surface, (0, 0))
    pygame.display.update()
    time.sleep(2)
    pygame.quit()
    return


def get_black_surface():

    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
 
    pygame.init()
    height_surface =  pygame.surfarray.make_surface(np.transpose(array, (1, 0, 2)))
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Height Surface Viewer")
    screen.blit(height_surface, (0, 0))
    pygame.display.update()
    clock = pygame.time.Clock()
    start = time.time()
    while time.time() - start < 5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(60)
    pygame.quit()
    return

'''
def make_line_surface(array): # turns all binary 1's too black
    height, width = array.shape
    
    # NOTE: Keep as width x height to match pygame
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[array == 1] = [255, 150, 0]  
    rgb_array[array < 1] = [0, 150, 255] 
    
    return pygame.surfarray.make_surface(rgb_array)
'''

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


def add_burnline():
    add_mitigation(BurnStatus.FIRELINE)
    return

def add_scratchline():
    add_mitigation(BurnStatus.SCRATCHLINE)
    return

def add_wetline():
    add_mitigation(BurnStatus.WETLINE)
    return

def add_mitigation(mitigation_type):
    # GET HxW INPUT ARRAY W 1 FOR LINE 0 FOR NO INPUT
    line_input = read_line(mitigation_type)
    line_input_points  = np.argwhere(line_input == 1) # [[row, col]]
    line_input_points = [tuple(pair) for pair in line_input_points] # [(row, col)]
    mitigations = [(*coord, mitigation_type) for coord in line_input_points]

    #mitigation_surface = make_line_surface(line_input)

    global sim
    #sim._blit_surface(mitigation_surface)
    sim.update_mitigation(mitigations)


def create_sim():

    config = Config("configs/camera_config.yml")
    print("configging")
    sim = FireSimulation(config)

    sim.rendering = True

    return sim

def playpause_sim():
    
    global running
    if running:
        running = False
    else:
        running = True
    print('play/paused')

    return

def start_sim():

    print('start_sim')
    
    sim = create_sim()
    
    global running
    running = True

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

            #mitigation_surface = make_line_surface(line_input)

            #sim._blit_surface(mitigation_surface)
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

def wait_for_pause():
    global exit
    playpause = input("Press Enter to play/pause the simulation or type 'exit' to exit: ").lower()
    if (playpause == 'exit') | (playpause == 'x'):
        exit = True
        global running
        running = False
    else:
        playpause_sim()
# -------------
# GUI COMPONENTS
# -------------

# -------------
# END GUI COMPONENTS
# -------------

# -------------
# REALSENSE COMPONENTS
# -------------


# -------------
# END REALSENSE COMPONENTS
# -------------

sim = initialize()

exit = False
while not exit:
    for event in pygame.event.get():
        print(event.type)
        if event.type == pygame.QUIT:
            print('quit')
            running = False
            exit = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
                exit = True

            elif event.key == pygame.K_SPACE:
                if running:
                    running = False
                    print('Simulation Paused')

                else:
                    running = True
                    print('Simulation Restarted')
                    
    if running:
        sim.run('1m')

sim.save_gif()

