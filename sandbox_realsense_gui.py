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
import tkinter as tk
import threading
import os

HEIGHT = 720
WIDTH = 1280
BASE_HEIGHT = 0.
MAX_HEIGHT = 2500.
#Color Constants from Red to Blue
#COLOR_CONSTANTS = [(0, 0, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150), (0, 255, 0), (150, 255, 0), (255, 255, 0), (255, 150, 0), (255, 0, 0)]
# Earth Tones
COLOR_CONSTANTS =  [(180, 160, 140), (180, 150, 100), (150, 150, 70), (130, 150, 70), (80, 130, 50), (40, 110, 70), (30, 100, 80), (30, 70, 100), (10, 20, 80)]

sim = None
running = False
calibrated = False
terrain_scanned = False
objects_scanned = False

def run_sim_loop():
    global running
    while running:
        sim.run('5m')

    return

def calibrate():
    print('calibrate/cancel')
    btn_Terrain.config(state='normal')
    btn_Objects.config(state='normal')

    #os.system("python calibrate.py")
    return

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


def scan_terrain():
    print('scan_terrain')
    global terrain_scanned
    global objects_scanned
    terrain_scanned = True
    if objects_scanned:
        btn_StartSim.config(state='normal')

    #os.system("python mapping.py")

    pygame.init()
    height_surface = get_height_surface(np.load("./data/depth_camera_input.npy"))
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Height Surface Viewer")
    screen.blit(height_surface, (0, 0))
    pygame.display.update()

    return

def scan_objects():
    print('scan_objects')
    global terrain_scanned
    global objects_scanned
    objects_scanned = True
    if terrain_scanned:
        btn_StartSim.config(state='normal')

    #os.system("python objects.py")

    return

def make_line_surface(array): # turns all binary 1's too black
    height, width = array.shape
    
    # NOTE: Keep as width x height to match pygame
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[array == 1] = [255, 150, 0]  
    rgb_array[array < 1] = [0, 150, 255] 
    
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


def add_mitigation_choices():
    print('add_mitigation')
    btn_Calibrate.grid_remove()
    btn_Terrain.grid_remove()
    btn_Objects.grid_remove()
    btn_Cancel.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=50, pady=5)
    btn_Burn.grid(row=1, column=0, sticky="nsew", padx=50, pady=5)
    btn_Scratch.grid(row=1, column=1, sticky="nsew", padx=50, pady=5)
    btn_Water.grid(row=1, column=2, sticky="nsew", padx=50, pady=5)
    return

def cancel_mitigation():

    print('calibrate/cancel')
    btn_Calibrate.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=50, pady=5)
    btn_Terrain.grid(row=1, column=0, sticky="nsew", padx=50, pady=5)
    btn_Objects.grid(row=1, column=2, sticky="nsew", padx=50, pady=5)
    btn_Cancel.grid_remove()
    btn_Burn.grid_remove()
    btn_Scratch.grid_remove()
    btn_Water.grid_remove()
    return

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

    mitigation_surface = make_line_surface(line_input)

    global sim
    sim._blit_surface(mitigation_surface)
    sim.update_mitigation(mitigations)


def create_sim():

    global sim
    config = Config("configs/camera_config.yml")

    sim = FireSimulation(config)

    sim.rendering = True

    btn_StartSim.grid_remove()
    btn_EndSim.grid(row=2, column=0, sticky="nsew", padx=50, pady=50)
    btn_PauseSim.grid(row=2, column=1, sticky="nsew", padx=50, pady=50)
    btn_AddMit.grid(row=2, column=2, sticky="nsew", padx=50, pady=50)
    btn_EndSim.config(background='#F00')
    btn_AddMit['fg'] = '#00F'
    btn_Calibrate.config(state='disabled')
    btn_Objects.config(state='disabled')
    btn_Terrain.config(state='disabled')
    threading.Thread(target=run_sim_loop, daemon=True).start()

    return

def playpause_sim():
    global running
    if running:
        running = False
        btn_AddMit.config(state='normal')
    else:
        running = True
        btn_AddMit.config(state='disabled')
        threading.Thread(target=run_sim_loop, daemon=True).start()
    print('play/paused')

    return

def start_sim():

    print('start_sim')
    btn_StartSim.grid_remove()
    btn_EndSim.grid(row=2, column=0, sticky="nsew", padx=50, pady=50)
    btn_PauseSim.grid(row=2, column=1, sticky="nsew", padx=50, pady=50)
    btn_AddMit.grid(row=2, column=2, sticky="nsew", padx=50, pady=50)
    btn_AddMit.config(state='disabled')
    btn_EndSim.config(background='#F00')
    btn_AddMit['fg'] = '#00F'
    btn_Calibrate.config(state='disabled')
    btn_Objects.config(state='disabled')
    btn_Terrain.config(state='disabled')
    
    global sim
    sim = create_sim()
    
    global running
    running = True
    threading.Thread(target=run_sim_loop, daemon=True).start()
    
    return

def end_sim():

    global running
    running = False
    btn_EndSim.grid_remove()
    btn_PauseSim.grid_remove()
    btn_AddMit.grid_remove()
    btn_StartSim.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=50, pady=50)
    btn_Calibrate.config(state='normal')
    btn_Objects.config(state='normal')
    btn_Terrain.config(state='normal')
    print('end_sim')

    global sim
    sim.save_gif()
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

# -------------
# GUI COMPONENTS
# -------------

gui = tk.Tk()
gui.title("AR Sandbox GUI")
gui.geometry("1200x800")

gui.grid_rowconfigure(0, weight=1)  
gui.grid_rowconfigure(1, weight=3) 
gui.grid_rowconfigure(2, weight=5)  

gui.grid_columnconfigure(0, weight=5)
gui.grid_columnconfigure(1, weight=5)
gui.grid_columnconfigure(2, weight=5)

# Top Section Calibrations
btn_Calibrate = tk.Button(gui, text="Calibrate", font=("Courier New", 20), command=calibrate)
btn_Terrain = tk.Button(gui, text="Scan Terrain", font=("Courier New", 20), command=scan_terrain)
btn_Objects = tk.Button(gui, text="Scan Objects", font=("Courier New", 20), command=scan_objects)

# Top Section Mitigations
btn_Cancel = tk.Button(gui, text="Cancel Mitigation", font=("Courier New", 20), command=cancel_mitigation)
btn_Water = tk.Button(gui, text="Water Line", font=("Courier New", 20), command=add_wetline)
btn_Burn = tk.Button(gui, text="Burn Line", font=("Courier New", 20), command=add_burnline)
btn_Scratch = tk.Button(gui, text="Scratch Line", font=("Courier New", 20), command=add_scratchline)

# Lower Section Start
btn_StartSim = tk.Button(gui, text="Start Simulation", font=("Courier New", 20), command=start_sim)
btn_StartSim.config(state='disabled')
btn_Terrain.config(state='disabled')
btn_Objects.config(state='disabled')

# Lower Section Controls
btn_EndSim = tk.Button(gui, text="End Simulation", font=("Courier New", 20), command=end_sim)
btn_PauseSim = tk.Button(gui, text="Pause Simulation", font=("Courier New", 20), command=playpause_sim)
btn_AddMit = tk.Button(gui, text="Add Mitigation", font=("Courier New", 20), command=add_mitigation_choices)


btn_Calibrate.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=50, pady=5)
btn_Terrain.grid(row=1, column=0, sticky="nsew", padx=50, pady=5)
btn_Objects.grid(row=1, column=2, sticky="nsew", padx=50, pady=5)
btn_StartSim.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=50, pady=50)

# -------------
# END GUI COMPONENTS
# -------------

# -------------
# REALSENSE COMPONENTS
# -------------


# -------------
# END REALSENSE COMPONENTS
# -------------

#model = YOLO("runs/detect/train3/weights/best.pt")#YOLO('yolov8n.pt') # runs/detect/train/weights/best.pt

#np.save("./data/depth_camera_input.npy", depth_array)

# TODO: ADD COORDINATES OF TREES
coordinates = []
np.save('./data/tree_coordinates.npy', coordinates)

gui.mainloop()

