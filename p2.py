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
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

# layer_outputs=[]
# for neuron_weights,neuron_bias in zip(weights,biases):
#     neuron_output =0
#     print(neuron_weights)
#     print(neuron_bias)
#     for n_input,weight in zip(inputs,neuron_weights):
#         neuron_output+=n_input*weight
#     neuron_output+=neuron_bias
#     layer_outputs.append(neuron_output)

some_value=0.5
weight=-0.7
bias=0.7

output=np.dot(inputs,np.array(weights).T)+biases
print(output)