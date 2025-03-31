import numpy as np

data = np.loadtxt('depthdata.txt')  # or use np.genfromtxt for more complex files
print(data.shape)
