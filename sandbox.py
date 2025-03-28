import pygame
import numpy as np
import threading
from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus

HEIGHT = 1080
WIDTH = 1440
BASE_HEIGHT = 100.
MAX_HEIGHT = 90.
#Color Constants from Red to Blue
COLOR_CONSTANTS = [(255, 0, 0), (255, 150, 0), (255, 255, 0), (150, 255, 0), (0, 255, 0), (0, 255, 150), (0, 255, 255), (0, 150, 255), (0, 0, 255)]

def get_height_surface(array):

    height, width = array.shape
    array = np.clip(array, BASE_HEIGHT, MAX_HEIGHT)
    bins = np.linspace(BASE_HEIGHT, MAX_HEIGHT, num=10)
    categories = np.digitize(array, bins) - 1 

    rgb_array = np.array([COLOR_CONSTANTS[c] for c in categories.flatten()])
    rgb_array = rgb_array.reshape(*array.shape, 3).astype(np.uint8)
    surface = pygame.Surface((width, height)) 
    pygame.surfarray.blit_array(surface, np.transpose(rgb_array, (1, 0, 2)))

    return surface

def make_line_surface(array):
    height, width = array.shape
    surface = pygame.Surface((width, height))  
    
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[array == 1] = [255, 0, 0]  
    rgb_array[array < 1] = [255, 255, 255] 
    
    pygame.surfarray.blit_array(surface, rgb_array)
    return surface

def read_line(mitigation_type):

    if mitigation_type not in [BurnStatus.FIRELINE, BurnStatus.SCRATCHLINE, BurnStatus.WETLINE]:
        print("Invalid mitigation type")
        return None
    
    # HAVE A CNN OR SOMETHING TO READ THE REALSENSE CAMERA INTO A 0/1 ARRAY FOR THE LINE
    # line_input = read_and_display_input()
    # TEST ARRAY
    line_input = np.zeros(HEIGHT, WIDTH, dtype=int)
    line_input[HEIGHT // 2, :] = 1
    return line_input


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

def create_sim():
    config = Config("configs/manual_config.yml")

    sim = FireSimulation(config)

    sim.rendering = True

    return sim


def run_simulation():

    global user_input
    user_input = False

    sim = create_sim()
    input_thread = threading.Thread(target=listen_for_playpause, daemon=True)
    input_thread.start()

    for i in range (10):
        sim.run("10m")
        print(f"Sim Step {i}")
        input_flag = input("1: PLAY SIM, 2: ADD MITIGATION, 3: DISPLAY CONTOURS, 4 EXIT\n")
        if input_flag == "1":
            continue
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
            
            sim.update_mitigation(mitigations)
        elif input_flag == "3":
            row_values = np.linspace(100, 90, WIDTH)
            height_array = np.tile(row_values, (HEIGHT, 1))
            height_surface = get_height_surface(height_array)

        else:

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

    user_input = None
    input_thread.join()

run_simulation()