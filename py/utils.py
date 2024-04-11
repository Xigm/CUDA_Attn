import torch

def write2file(path, inputs, W_q, W_k, W_v, dk, n_tokens):
    with open(path, "w") as f:
        string2write = prepare_string(inputs, W_q, W_k, W_v, dk, n_tokens)
        f.write(string2write)

def readfromfile(path, dk, n_tokens):
    outputs = torch.zeros(n_tokens, dk)

    with open(path, "r") as f:
        for i in range(n_tokens):
            for j in range(dk):
                outputs[i][j] = float(f.readline())
        
    return outputs

def prepare_string(inputs, W_q, W_k, W_v, dk, n_tokens):
    string2write = ""
    for i in range(n_tokens):
        for j in range(dk):
            string2write += str(inputs[i][j].item()) + " "
        string2write += "\n"
    
    for i in range(dk):
        for j in range(dk):
            string2write += str(W_q[i][j].item()) + " "
        string2write += "\n"
    
    for i in range(dk):
        for j in range(dk):
            string2write += str(W_k[i][j].item()) + " "
        string2write += "\n"
    
    for i in range(dk):
        for j in range(dk):
            string2write += str(W_v[i][j].item()) + " "
        string2write += "\n"
    
    return string2write