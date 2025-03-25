import pygame
import numpy as np
from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus

HEIGHT = 1080
WIDTH = 1440


def make_line_surface(array):
    height, width = array.shape
    surface = pygame.Surface((width, height))  
    
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[array == 1] = [0, 0, 255]  
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
    line_input = np.zeros(HEIGHT, WIDTH)
    line_input[0:10,0] = 1
    line_input_points  = np.argwhere(line_input == 1) # [[row, col]]
    line_input_points = [tuple(pair) for pair in line_input_points] # [(row, col)]
    mitigation_input_points = [(*coord, mitigation_type) for coord in line_input_points] # [(row, col, mitigation_type)]

    return mitigation_input_points

def create_sim():
    config = Config("configs/manual_config.yml")

    sim = FireSimulation(config)

    sim.rendering = True

    return sim


def main():
    sim = create_sim()

    for i in range (10):
        sim.run("10m")

        if False: # UPDATE WITH A WAY TO PAUSE/PLAY
            if False: # UPDATE WITH A WAY TO CHOOSE TO ADD A MITIGATION
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
                read_line(mitigation_type)
                sim.update_mitigation()

        