import numpy as np

file_path = "data/debug_softmax.txt"
data_np = np.zeros(32768)

with open(file_path, "r") as file:
    data = file.read().splitlines()

# pass data to the data_np 
for i in range(32768):
    data_np[i] = float(data[i])

# print the data
print(data_np)

# print the max value
print("Max value: ", data_np.max())

print(np.exp(data_np.max()))