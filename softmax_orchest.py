from py.attention_python import simple_self_attention, gpu_self_attention, gpu_self_attention_timed
from py.utils import write2file_sm, readfromfile
import torch
import os
import time
import matplotlib.pyplot as plt

FILES_COMPILE = [ 
                  'main_softmax.cu',
                ]

C_PATH = "./c/"
CU_PATH = "./cuda/"

# add path to the files
FILES_COMPILE = [C_PATH + file if file.endswith('.c') else CU_PATH + file for file in FILES_COMPILE ]

output_file = "./bin/main_sm"
data_input = "./data/inputs_sm.txt"
data_output = "./data/outputs_sm.txt"

# dk = 32*2
# n_tokens = 32*1
# n_heads = 16
# gpt2 sizes : 768, 1024, 1200, 1600
dk = 4
n_tokens = 33
n_heads = 2
head_dim = dk // n_heads
batch_size = 2

# torch.manual_seed(2026)

# define the input tensor
inputs = torch.randn((batch_size, n_heads, n_tokens, n_tokens), dtype=torch.float32).masked_fill(torch.tril(torch.ones((n_tokens, n_tokens))) == 0, float('-inf'))
# inputs[0,0,-1,-1] = 2

weights = inputs

# write data
write2file_sm(data_input, weights, n_tokens, batch_size, n_heads)

# call the function
time_start = time.time()
output_py = torch.nn.functional.softmax(inputs, dim = -1)
time_taken = time.time() - time_start

# Send the input tensors to the GPU
inputs_c = inputs.cuda()

time_start = time.time()
output_py_gpu = torch.nn.functional.softmax(inputs_c, dim = -1)
time_taken_gpu = time.time() - time_start

# print(inputs[0,0,-1,:].max())
print(sum(torch.exp(inputs[0,1,0,:] - inputs[0,1,0,:].max())))
print(torch.exp(inputs[0,1,0,:] - inputs[0,1,0,:].max()))
print(inputs[0,1,0,:].max())
print(output_py[0,1,0,:])

del inputs_c

print("Time taken by python code: ", time_taken)
print("Time taken by py-cuda code: ", time_taken_gpu)

warning_supression = " -diag-suppress 549"
# warning_supression = ""
# compile main.c with nvcc using os
command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE) + warning_supression
os.system(command)

# run the compiled program
os.system(str(output_file) + ' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(n_heads) + ' ' + str(batch_size))

output_c, time_taken, time_taken_k = readfromfile(data_output, n_tokens, n_tokens, batch_size, n_heads)

# compare the results if output_py and output_c are the same and print true if they are
if torch.allclose(output_py, output_c, atol=1e-04):
    print("Implementation is correct!")
else:
    print("Implementation is incorrect!")

# compute tolerance
tolerance = (output_py - output_c).abs().max().item()
print("Tolerance: ", tolerance)




