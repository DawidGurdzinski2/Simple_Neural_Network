import sys
import numpy as np
import matplotlib

# print("Python",sys.version)
# print("numpy:",np.__version__)
# print("Matplotlib: ", matplotlib.__version__)

inputs = [[0.2, 0.8, -0.5, 1.0],
          [0.2, 0.8, -0.5, 1.0],
          [0.2, 0.8, -0.5, 1.0]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.226, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]



weights2 = [[0.2, 0.8, -0.5,],
           [0.5, -0.91, 0.26],
           [-0.26, -0.27, 0.17]]

biases2 = [2.0, 3.0, 0.5]




layer1_outputs=np.dot(inputs,np.array(weights).T)+biases
print(layer1_outputs)
layer2_outputs=np.dot(layer1_outputs,np.array(weights2).T)+biases2


print(layer2_outputs)