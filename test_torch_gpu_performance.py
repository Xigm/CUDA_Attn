import torch
import time
from tqdm import tqdm

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Create large random matrices
matrix_size = (10000, 10000)


# Number of times to perform matrix multiplication
num_iterations = 10

# Perform matrix multiplication on GPU and measure time
individual_times = []
for _ in tqdm(range(num_iterations)):
    matrix_a = torch.randn(matrix_size).to(device)
    matrix_b = torch.randn(matrix_size).to(device)
    start_time = time.time()
    result = torch.matmul(matrix_a, matrix_b)
    end_time = time.time()
    individual_times.append(end_time - start_time)

# Move the result back to CPU if needed
result = result.cpu()

# Print the result
print(result)

# Compute and print the average time taken
average_time = (end_time - start_time) / num_iterations
print(f"Average time taken: {average_time} seconds")
print("Individual times:")
for i in range(num_iterations):
    print(f"Iteration {i+1}: {end_time - start_time} seconds")