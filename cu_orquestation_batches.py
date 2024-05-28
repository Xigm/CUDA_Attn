from py.attention_python import simple_self_attention, gpu_self_attention, gpu_self_attention_timed
from py.utils import write2file, readfromfile
import torch
import os
import time

FILES_COMPILE = [ 
                  'main_batches.cu',
                ]

C_PATH = "./c/"
CU_PATH = "./cuda/"

# add path to the files
FILES_COMPILE = [C_PATH + file if file.endswith('.c') else CU_PATH + file for file in FILES_COMPILE ]

output_file = "./bin/main_batches"
data_input = "./data/inputs_batches.txt"
data_output = "./data/outputs_batches.txt"

# dk = 32*2
# n_tokens = 32*1
# n_heads = 16
# gpt2 sizes : 768, 1024, 1200, 1600
dk = 32
n_tokens = 32 
n_heads = 2
head_dim = dk // n_heads
batch_size = 32

torch.manual_seed(2026)

# define the input tensor
inputs = torch.randn((batch_size, n_tokens, dk), dtype=torch.float32)

# define the query, key, and value projection matrices
W_q = torch.randn((dk, dk), dtype=torch.float32)*0.5
W_q = W_q*torch.abs(W_q)
W_k = torch.randn((dk, dk), dtype=torch.float32)*0.5
W_k = W_k*torch.abs(W_k)
W_v = torch.randn((dk, dk), dtype=torch.float32)*0.5
W_v = W_v*torch.abs(W_v)
W_cproj = torch.randn((dk, dk), dtype=torch.float32)*0.5
W_cproj = W_cproj*torch.abs(W_cproj)

import matplotlib.pyplot as plt

# Plot the probability distributions of the weight matrices
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].hist(W_q.flatten().cpu().numpy(), bins=50, density=True)
axs[0, 0].set_title('W_q')
axs[0, 1].hist(W_k.flatten().cpu().numpy(), bins=50, density=True)
axs[0, 1].set_title('W_k')
axs[1, 0].hist(W_v.flatten().cpu().numpy(), bins=50, density=True)
axs[1, 0].set_title('W_v')
axs[1, 1].hist(W_cproj.flatten().cpu().numpy(), bins=50, density=True)
axs[1, 1].set_title('W_cproj')

plt.tight_layout()
plt.savefig("./results/weights_distributions.png")

# print max of weights
print("Max of W_q: ", W_q.max().item())
print("Max of W_k: ", W_k.max().item())
print("Max of W_v: ", W_v.max().item())
print("Max of W_cproj: ", W_cproj.max().item())


weights = [inputs, W_q, W_k, W_v, W_cproj]

# write data
write2file(data_input, weights, dk, n_tokens, batch_size)

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
os.system(str(output_file) +' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(dk) + ' ' + str(batch_size))

output_c, time_taken = readfromfile(data_output, dk, n_tokens, batch_size)

# compare the results if output_py and output_c are the same and print true if they are
if torch.allclose(output_py, output_c, atol=1e-04):
    print("Implementation is correct!")
else:
    print("Implementation is incorrect!")

# compute tolerance
tolerance = (output_py - output_c).abs().max().item()
print("Tolerance: ", tolerance)


