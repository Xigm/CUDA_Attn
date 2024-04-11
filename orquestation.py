from attention_python import simple_self_attention
from utils import write2file, readfromfile
import torch
import os

FILES_COMPILE = [ 
                  'attention.c',
                  'matmul.c',
                  'softmax.c',
                  'main.c',
                  'transpose.c',
                  
                  ]

output_file = "main"
data_input = "./data/inputs.txt"
data_output = "./data/outputs.txt"

dk = 4
n_tokens = 2
n_heads = 2
head_dim = dk // n_heads

torch.manual_seed(2024)

# define the input tensor
inputs = torch.randint(0, 10,(n_tokens, dk), dtype=torch.float32)

# define the query, key, and value projection matrices
W_q = torch.randint(0, 10, (dk, dk), dtype=torch.float32)
W_k = torch.randint(0, 10, (dk, dk), dtype=torch.float32)
W_v = torch.randint(0, 10, (dk, dk), dtype=torch.float32)

# write data
write2file(data_input, inputs, W_q, W_k, W_v, dk, n_tokens)

# call the function
output_py = simple_self_attention(inputs, W_q, W_k, W_v, dk)

# compile main.c with nvcc using os
command = "nvcc -o " + str(output_file) + " " + " ".join(FILES_COMPILE)
os.system(command)

# run the compiled program
os.system('./' + str(output_file) +' ' + data_input + ' ' + data_output + ' '+ str(n_tokens)  + ' ' + str(dk))

output_c = readfromfile(data_output, dk, n_tokens)

# compare the results if output_py and output_c are the same and print true if they are
if torch.allclose(output_py, output_c):
    print("Implementation is correct!")


