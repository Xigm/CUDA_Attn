from py.attention_python import simple_self_attention, gpu_self_attention, gpu_self_attention_timed
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
dk = 64
n_tokens = 128
n_heads = 2
head_dim = dk // n_heads

times_cpu = []
times_gpu = []
times_c = []
dks = [64, 128, 256, 512, 768]
dks = [64, 128, 256]
n_tokenss = [128, 256, 512, 1024]
for n_tokens in n_tokenss:

    # define the input tensor
    inputs = torch.randn((n_tokens, dk), dtype=torch.float32)

    # define the query, key, and value projection matrices
    W_q = torch.randn((dk, dk), dtype=torch.float32)*0.01
    W_k = torch.randn((dk, dk), dtype=torch.float32)*0.01
    W_v = torch.randn((dk, dk), dtype=torch.float32)*0.01
    W_cproj = torch.randn((dk, dk), dtype=torch.float32)*0.01

    weights = [inputs, W_q, W_k, W_v, W_cproj]

    # write data
    write2file(data_input, weights, dk, n_tokens, None)

    # call the function
    time_start = time.time()
    output_py = simple_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk)
    time_taken = time.time() - time_start
    times_cpu.append(time_taken)

    # Send the input tensors to the GPU
    torch.cuda.empty_cache()
    inputs_c = inputs.cuda()
    W_q_c = W_q.cuda()
    W_k_c = W_k.cuda()
    W_v_c = W_v.cuda()
    W_cproj_c = W_cproj.cuda()
    mask = torch.tril(torch.ones((n_tokens, n_tokens), device = "cuda")) == 0

    time_start = time.time()
    output_py_gpu = gpu_self_attention_timed(inputs_c, W_q_c, W_k_c, W_v_c, W_cproj_c, dk, mask)
    time_taken_gpu = time.time() - time_start
    output_py_gpu = output_py_gpu.cpu()
    times_gpu.append(time_taken_gpu)

    print("Time taken by python code: ", time_taken)
    print("Time taken by py-cuda code: ", time_taken_gpu)

    warning_supression = " -diag-suppress 549"
    # warning_supression = ""
    # compile main.c with nvcc using os
    command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE) + warning_supression
    os.system(command)

    # run the compiled program
    os.system(str(output_file) +' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(dk))

    output_c, time_taken_c = readfromfile(data_output, dk, n_tokens)
    times_c.append(time_taken_c)

    # compare the results if output_py and output_c are the same and print true if they are
    if torch.allclose(output_py, output_c, atol=1e-04):
        print("Implementation is correct!")
    else:
        print("Implementation is incorrect!")

    # compute tolerance
    tolerance = (output_py - output_c).abs().max().item()
    print("Tolerance: ", tolerance)

# plot times with legend
import matplotlib.pyplot as plt
plt.plot(n_tokenss, times_cpu, label="CPU")
plt.plot(n_tokenss, times_gpu, label="GPU")
plt.plot(n_tokenss, times_c, label="CUDA")
plt.xlabel("dk")
plt.ylabel("Time taken")
plt.legend()
plt.savefig("./results/performance"+str(n_tokens)+str(dks)+".png")

