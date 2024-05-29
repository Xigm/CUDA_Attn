from py.attention_python import simple_self_attention, gpu_self_attention, gpu_self_attention_timed
from py.utils import write2file_sm, readfromfile
import torch
import os
import time
import matplotlib.pyplot as plt
import json

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
path_to_save_dict = "./dict_softmax_fix.json"
path_to_save = "./softmax_times_fix.png"

# dk = 32*2
# n_tokens = 32*1
# n_heads = 16
# gpt2 sizes : 768, 1024, 1200, 1600
dk = 1024
n_heads = 1
head_dim = dk // n_heads
batch_size = 1

# torch.manual_seed(2026)

# with open(path_to_save_dict, 'r') as f:
#     values_tried = json.load(f)

values_tried = {}

while len(values_tried.keys()) != 1024:

    # get random value from 2 to 1024 
    n_tokens = torch.randint(2, 1025, (1,)).item()

    while n_tokens in values_tried.keys():
        n_tokens = torch.randint(2, 1025, (1,)).item()

    # define the input tensor
    inputs = torch.randn((batch_size, n_heads, n_tokens, n_tokens), dtype=torch.float32).masked_fill(torch.tril(torch.ones((n_tokens, n_tokens))) == 0, float('-inf'))
    inputs[0,0,-1,-1] = 2

    weights = inputs

    # write data
    write2file_sm(data_input, weights, n_tokens, batch_size, n_heads)

    # call the function
    time_start = time.time()
    with torch.no_grad():
        output_py = torch.nn.functional.softmax(inputs, dim = -1)
    time_taken = time.time() - time_start

    # Send the input tensors to the GPU
    inputs.to("cuda")

    time_start = time.time()
    with torch.no_grad():
        output_py_gpu = torch.nn.functional.softmax(inputs, dim = -1)
    time_taken_gpu = time.time() - time_start

    print("Time taken by python code: ", time_taken)
    print("Time taken by py-cuda code: ", time_taken_gpu)

    warning_supression = " -diag-suppress 549"
    # warning_supression = ""
    # compile main.c with nvcc using os
    command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE) + warning_supression
    os.system(command)

    # run the compiled program
    os.system(str(output_file) + ' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(n_heads) + ' ' + str(batch_size))

    output_c, time_taken_c, time_taken_kernel = readfromfile(data_output, n_tokens, n_tokens, batch_size, n_heads)

    values_tried[n_tokens] = {'py' : time_taken, 'py_gpu' : time_taken_gpu, 'c' : time_taken_c, 'c_kernel' : time_taken_kernel}

    # plot all values_tried and save them in path.fig in logx scale

    # Extract data for plotting
    n_tokens = list(values_tried.keys())
    py_times = [values_tried[n]['py'] for n in n_tokens]
    py_gpu_times = [values_tried[n]['py_gpu'] for n in n_tokens]
    c_times = [values_tried[n]['c'] for n in n_tokens]
    c_kernel_times = [values_tried[n]['c_kernel'] for n in n_tokens]

    # sort values by n_tokens
    n_tokens, indexes = torch.sort(torch.tensor(n_tokens))
    py_times = torch.tensor(py_times)[indexes].tolist()
    py_gpu_times = torch.tensor(py_gpu_times)[indexes].tolist()
    c_times = torch.tensor(c_times)[indexes].tolist()
    c_kernel_times = torch.tensor(c_kernel_times)[indexes].tolist()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_tokens, py_times, label='Python', marker='o')
    plt.plot(n_tokens, py_gpu_times, label='Python GPU', marker='o')
    # plt.plot(n_tokens, c_times, label='C', marker='o')
    plt.plot(n_tokens, c_kernel_times, label='Cuda Kernel', marker='o')

    # Logarithmic scale for x-axis
    # plt.xscale('log')

    # Adding titles and labels
    plt.title('Performance Comparison')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Time Taken (s)')
    plt.legend()
    plt.grid(True)

    # set x max at 0.002
    plt.ylim(0, 0.002)

    # Save the figure
    plt.savefig(path_to_save)

    # save dict
    with open(path_to_save_dict, 'w') as f:
        json.dump(values_tried, f, indent=4)






