from py.attention_python import simple_self_attention, gpu_self_attention, MH_self_attention
from py.utils import write2file, readfromfile, read_matrix_from_file, read_matrix_from_file_sm
import torch
import os
import time
import matplotlib.pyplot as plt


FILES_COMPILE = [ 
                  'main_MHA_benchmarks.cu',
                ]

C_PATH = "./c/"
CU_PATH = "./cuda/"

# add path to the files
FILES_COMPILE = [C_PATH + file if file.endswith('.c') else CU_PATH + file for file in FILES_COMPILE ]

output_file = "./bin/main_bench"
# data_input = "./data/inputs_MHA.txt"
data_output = "./data/outputs_bench.txt"

# dk = 32*2 
# n_tokens = 32*1
# n_heads = 16
# gpt2 sizes : 768, 1024, 1200, 1600
# max dk now is 1024
dk = 1200
n_tokens = 1024
n_heads = 12
head_dim = dk // n_heads
batch_size = 4

torch.manual_seed(2026)

times_py, times_gpu, times_kernel = [],[],[]

dks = torch.arange(120,1200,120, requires_grad=False)

dks = torch.cat([torch.tensor(12).view(1), dks])

for dk in dks:
    dk = dk.item()

    # define the input tensor
    inputs = torch.randn((batch_size, n_tokens, dk), dtype=torch.float32)

    mod_factor = 0.5
    # define the query, key, and value projection matrices
    W_q = torch.randn((dk, dk), dtype=torch.float32)*mod_factor
    W_q = W_q*torch.abs(W_q)
    W_k = torch.randn((dk, dk), dtype=torch.float32)*mod_factor
    W_k = W_k*torch.abs(W_k)
    W_v = torch.randn((dk, dk), dtype=torch.float32)*mod_factor
    W_v = W_v*torch.abs(W_v)
    W_cproj = torch.randn((dk, dk), dtype=torch.float32)*mod_factor
    W_cproj = W_cproj*torch.abs(W_cproj)


    # print max of weights
    # print("Max of W_q: ", W_q.max().item())
    # print("Max of W_k: ", W_k.max().item())
    # print("Max of W_v: ", W_v.max().item())
    # print("Max of W_cproj: ", W_cproj.max().item())


    weights = [inputs, W_q, W_k, W_v, W_cproj]

    # write data
    # write2file(data_input, weights, dk, n_tokens, batch_size)

    # call the function
    time_start = time.time()
    with torch.no_grad():
        output_py, debug_info = MH_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk, n_tokens, n_heads, batch_size)
    time_taken = time.time() - time_start

    # Send the input tensors to the GPU 
    inputs_c = inputs.cuda()
    W_q_c = W_q.cuda()
    W_k_c = W_k.cuda()
    W_v_c = W_v.cuda()
    W_cproj_c = W_cproj.cuda()
    mask = torch.tril(torch.ones((batch_size, n_heads, n_tokens, n_tokens), device = "cuda")) == 0

    time_start = time.time()
    with torch.no_grad():
        output_py_gpu, _ = MH_self_attention(inputs_c, W_q_c, W_k_c, W_v_c, W_cproj_c, dk, n_tokens, n_heads, batch_size, mask = mask)
    time_taken_gpu = time.time() - time_start

    # print("Time taken by python code: ", time_taken)
    # print("Time taken by py-cuda code: ", time_taken_gpu)

    warning_supression = " -diag-suppress 549"
    # warning_supression = ""
    # compile main.c with nvcc using os
    command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE) + warning_supression
    os.system(command)

    # run the compiled program
    os.system(str(output_file) + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(dk) + ' ' + str(batch_size) + ' ' + str(n_heads))

    # output_c, time_taken, time_taken_kernel = readfromfile(data_output, dk, n_tokens, batch_size)
    # output_debug_c = read_matrix_from_file("./data/debug_output.txt", n_tokens, dk, batch_size)
    # output_debug_c = read_matrix_from_file_sm("./data/debug_softmax.txt", n_tokens, n_tokens, n_heads, batch_size)

    # read kernel time from output file
    
    # read kernel time from output file
    with open(data_output, 'r') as f:
        lines = f.readlines()
        time_taken_kernel = float(lines[-1].strip())

    # print("Kernel time taken: ", time_taken_kernel)

    times_py.append(time_taken*1000)
    times_gpu.append(time_taken_gpu*1000)
    times_kernel.append(time_taken_kernel)


# plot times
# plt.plot(dks, times_py, label = "Plain torch")
plt.plot(dks[1:], times_gpu[1:], label = "GPU torch")
plt.plot(dks[1:], times_kernel[1:], label = "Cuda custom kernel")
plt.legend()
plt.show()


