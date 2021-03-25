import numpy as np

inputs = [[1, 2, 3, 4],
          [4.6,4.4,0.5,1.1],
          [-0.7,-0.4,-8.5,-3.4]]

weights1 = [[3.1, 6.3, -0.4, 1.0],
           [0.5, -6.3, -0.4, 6.0],
           [-2.0, -8.3, 0.4,0.7] ]

biases1 = [2, 3, 0.5]

weights2 = [[9.9, -0.34, -0.4],
           [-0.99, 0.13, -5.3],
           [-2.0, -3.5, -0.5] ]

biases2 = [5.4, 0.5, 5.5]

layer_1_output = np.dot(inputs,np.array(weights1).T) + biases1
layer_2_output = np.dot(layer_1_output,np.array(weights2).T) + biases2
print(layer_2_output)


'''
output_value = []
for neuron_weight,neuron_bias in zip(weights,biases):
    neuron_output = 0
    for weight,neuron_input in zip(neuron_weight,inputs):
        neuron_output += weight * neuron_input
    neuron_output += neuron_bias
    output_value.append(neuron_output)

print(output_value)
'''
