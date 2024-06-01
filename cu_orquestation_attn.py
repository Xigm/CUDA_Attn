from py.attention_python import simple_self_attention, gpu_self_attention
from py.utils import write2file, readfromfile
import torch
import os
import time

FILES_COMPILE = [ 
                  'main.cu',
                ]

C_PATH = "./c/"
CU_PATH = "./cuda/"

# add path to the files
FILES_COMPILE = [C_PATH + file if file.endswith('.c') else CU_PATH + file for file in FILES_COMPILE ]

output_file = "./bin/main"
data_input = "./data/inputs.txt"
data_output = "./data/outputs.txt"

# dk = 32*2
# n_tokens = 32*1
# n_heads = 16
# gpt2 sizes : 768, 1024, 1200, 1600
dk = 768
n_tokens = 64
n_heads = 12
head_dim = dk // n_heads

# torch.manual_seed(2026)

# define the input tensor
inputs = torch.randn((n_tokens, dk), dtype=torch.float32)

# define the query, key, and value projection matrices
W_q = torch.randn((dk, dk), dtype=torch.float32)*0.01
W_k = torch.randn((dk, dk), dtype=torch.float32)*0.01
W_v = torch.randn((dk, dk), dtype=torch.float32)*0.01
W_cproj = torch.randn((dk, dk), dtype=torch.float32)*0.01

weights = [inputs, W_q, W_k, W_v, W_cproj]

# write data
write2file(data_input, weights, dk, n_tokens)

# call the function
time_start = time.time()
output_py = simple_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk)
time_taken = time.time() - time_start



# Send the input tensors to the GPU
inputs_c = inputs.cuda()
W_q_c = W_q.cuda()
W_k_c = W_k.cuda()
W_v_c = W_v.cuda()
W_cproj_c = W_cproj.cuda()
mask = torch.tril(torch.ones((n_tokens, n_tokens), device = "cuda")) == 0

time_start = time.time()
output_py_gpu = gpu_self_attention(inputs_c, W_q_c, W_k_c, W_v_c, W_cproj_c, dk, mask)
time_taken_gpu = time.time() - time_start

print("Time taken by python code: ", time_taken)
print("Time taken by py-cuda code: ", time_taken_gpu)

warning_supression = " -diag-suppress 549"
# warning_supression = ""
# compile main.c with nvcc using os
command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE) + warning_supression
os.system(command)

# run the compiled program
os.system(str(output_file) +' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(dk))

output_c = readfromfile(data_output, dk, n_tokens)

# compare the results if output_py and output_c are the same and print true if they are
if torch.allclose(output_py, output_c, atol=1e-04):
    print("Implementation is correct!")
else:
    print("Implementation is incorrect!")

# compute tolerance
tolerance = (output_py - output_c).abs().max().item()
print("Tolerance: ", tolerance)


