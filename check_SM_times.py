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
path_to_save_dict = "./dict_softmax_many2_nh_12.json"
path_to_save = "./softmax_times_many2_nh_12.png"

# dk = 32*2
# n_tokens = 32*1
# n_heads = 12
# gpt2 sizes : 768, 1024, 1200, 1600
dk = 1024
n_heads = 12
head_dim = dk // n_heads
batch_size = 16

# torch.manual_seed(2026)

# with open(path_to_save_dict, 'r') as f:
#     values_tried = json.load(f)

values_tried = {}
repetitions = 5
# block_sizes = [8, 16, 32, 64]
# block_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
block_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 224, 256, 384, 512, 768, 1024]

for block_size in block_sizes:

    values_tried[block_size] = {'py' : 0, 'py_gpu' : 0, 'c_kernel' : 0}

    # get random value from 2 to 1024 
    n_tokens = block_size

    # define the input tensor
    inputs = torch.randn((batch_size, n_heads, n_tokens, n_tokens), dtype=torch.float32).masked_fill(torch.tril(torch.ones((n_tokens, n_tokens))) == 0, float('-inf'))
    inputs[0,0,-1,-1] = 2

    weights = inputs

    # write data
    write2file_sm(data_input, weights, n_tokens, batch_size, n_heads)

    # call the function
    time_taken = 0
    for i in range(repetitions):
        with torch.no_grad():
            time_start = time.time()
            output_py = torch.nn.functional.softmax(inputs, dim = -1)
            time_taken += time.time() - time_start

    time_taken = time_taken/repetitions

    # Send the input tensors to the GPU
    inputs.to("cuda")

    time_taken_gpu = 0
    for i in range(repetitions):
        with torch.no_grad():
            time_start = time.time()
            output_py_gpu = torch.nn.functional.softmax(inputs, dim = -1)
            time_taken_gpu += time.time() - time_start
    
    time_taken_gpu = time_taken_gpu/repetitions

    # print("Time taken by python code: ", time_taken)
    # print("Time taken by py-cuda code: ", time_taken_gpu)

    warning_supression = " -diag-suppress 549"
    # warning_supression = ""
    # compile main.c with nvcc using os
    command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE) + warning_supression
    os.system(command)

    time_taken_kernel = 0
    for i in range(repetitions):
    # run the compiled program
        os.system(str(output_file) + ' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(n_heads) + ' ' + str(batch_size))

        output_c, time_taken_c, time_taken_kernel_i = readfromfile(data_output, n_tokens, n_tokens, batch_size, n_heads)

        time_taken_kernel += time_taken_kernel_i

    time_taken_kernel = time_taken_kernel/repetitions

    values_tried[block_size] = {'py' : time_taken/10 + values_tried[block_size]['py'],
                                'py_gpu' : time_taken_gpu/10 + values_tried[block_size]['py_gpu'],
                                'c_kernel' : time_taken_kernel/10 + values_tried[block_size]['c_kernel']}

    # plot all values_tried and save them in path.fig in logx scale

    # Extract data for plotting
    n_tokens = list(values_tried.keys())
    py_times = [values_tried[n]['py'] for n in n_tokens]
    py_gpu_times = [values_tried[n]['py_gpu'] for n in n_tokens]
    c_kernel_times = [values_tried[n]['c_kernel'] for n in n_tokens]

    # sort values by n_tokens
    n_tokens, indexes = torch.sort(torch.tensor(n_tokens))
    py_times = torch.tensor(py_times)[indexes].tolist()
    py_gpu_times = torch.tensor(py_gpu_times)[indexes].tolist()
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
    plt.title('Performance Comparison batch ' + str(batch_size) + " n_heads " + str(n_heads))
    plt.xlabel('Number of Tokens')
    plt.ylabel('Time Taken (s)')
    plt.legend()
    plt.grid(True)

    # set x max at 0.002
    # plt.ylim(0, 0.002)

    # Save the figure
    plt.savefig(path_to_save)

    # save dict
    with open(path_to_save_dict, 'w') as f:
        json.dump(values_tried, f, indent=4)






