import numpy as np
import pygame

# Define constants
MIN_HT = 0   # Minimum height value
MAX_HT = 100 # Maximum height value

# Define 9 RGB colors corresponding to categories
COLOR_CONSTANTS = [
    (0, 0, 255),   # Blue
    (0, 128, 255), # Light Blue
    (0, 255, 255), # Cyan
    (0, 255, 128), # Turquoise
    (0, 255, 0),   # Green
    (255, 255, 0), # Yellow
    (255, 165, 0), # Orange
    (255, 69, 0),  # Red-Orange
    (255, 0, 0)    # Red
]

def array_to_rgb(array):
    """
    Converts an MxN NumPy array into an MxNx3 RGB array.
    
    :param array: NumPy array of shape (M, N) with values in the range [MIN_HT, MAX_HT]
    :return: NumPy array of shape (M, N, 3) with RGB values
    """
    # Clip values to be within the defined min/max range
    array = np.clip(array, MIN_HT, MAX_HT)

    # Compute category edges
    bins = np.linspace(MIN_HT, MAX_HT, num=10)  # 9 categories → 10 edges

    # Digitize array values into categories (values from 1 to 9)
    categories = np.digitize(array, bins) - 1  # Convert to 0-based indexing

    # Map categories to RGB colors
    rgb_array = np.array([COLOR_CONSTANTS[c] for c in categories.flatten()])
    return rgb_array.reshape(*array.shape, 3).astype(np.uint8)  # Convert back to (M, N, 3)

def display_surface(array):
    """
    Converts an MxN array to an RGB surface and displays it using Pygame.
    
    :param array: NumPy array of shape (M, N)
    """
    pygame.init()
    
    # Convert array to RGB
    rgb_array = array_to_rgb(array)

    # Get dimensions
    height, width = array.shape

    # Create Pygame surface
    #surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))  # Pygame uses (width, height)
    surface = pygame.Surface((width, height)) 
    pygame.surfarray.blit_array(surface, np.transpose(rgb_array, (1, 0, 2)))

    # Set up display
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Heightmap Visualization")

    # Display image
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Event loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

# Example Usage
if __name__ == "__main__":
    M, N = 200, 300  # Define size of the heightmap
    heightmap = np.random.uniform(MIN_HT, MAX_HT, size=(M, N))  # Generate random height values
    display_surface(heightmap)
