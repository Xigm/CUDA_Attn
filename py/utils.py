import torch

def write2file(path, weights, dk, n_tokens, batch_size):
    with open(path, "w") as f:
        string2write = prepare_string_V2(weights, dk, n_tokens, batch_size)
        f.write(string2write)

def write2file_sm(data_input, weights, n_tokens, batch_size, n_heads):
    with open(data_input, "w") as f:
        string2write = prepare_string_sm(weights, n_tokens, batch_size, n_heads)
        f.write(string2write)

def readfromfile(path, dk, n_tokens, batch_size = None, n_heads = None):
    if batch_size is None:
        outputs = torch.zeros(n_tokens, dk)

        with open(path, "r") as f:
            for i in range(n_tokens):
                for j in range(dk):
                    outputs[i][j] = float(f.readline())

            time_taken = float(f.readline())
            
        return outputs, time_taken
    
    else:
        if n_heads is None:
            outputs = torch.zeros(batch_size, n_tokens, dk)

            with open(path, "r") as f:
                for k in range(batch_size):
                    for i in range(n_tokens):
                        for j in range(dk):
                            outputs[k][i][j] = float(f.readline())

                time_taken = float(f.readline())

        else:
            outputs = torch.zeros(batch_size, n_heads, n_tokens, n_tokens)

            with open(path, "r") as f:
                for k in range(batch_size):
                    for h in range(n_heads):
                        for i in range(n_tokens):
                            for j in range(n_tokens):
                                outputs[k][h][i][j] = float(f.readline())

                time_taken = float(f.readline())

            
        return outputs, time_taken
        



def prepare_string(weights, dk, n_tokens):
    string2write = ""
    
    for i in range(n_tokens):
        for j in range(dk):
            string2write += str(weights[0][i][j].item()) + " "
        string2write += "\n"
    
    for i in range(dk):
        for j in range(dk):
            string2write += str(weights[1][i][j].item()) + " "
        string2write += "\n"
    
    for i in range(dk):
        for j in range(dk):
            string2write += str(weights[2][i][j].item()) + " "
        string2write += "\n"
    
    for i in range(dk):
        for j in range(dk):
            string2write += str(weights[3][i][j].item()) + " "
        string2write += "\n"

    for i in range(dk):
        for j in range(dk):
            string2write += str(weights[4][i][j].item()) + " "
        string2write += "\n"

    if len(weights) == 8:
        for i in range(dk):
            for j in range(dk):
                string2write += str(weights[4][i][j].item()) + " "
            string2write += "\n"
        
        for i in range(dk):
            for j in range(dk*4):
                string2write += str(weights[5][i][j].item()) + " "
            string2write += "\n"
        
        for i in range(dk*4):
            for j in range(dk):
                string2write += str(weights[6][i][j].item()) + " "
            string2write += "\n"
    
    return string2write

def format_matrix(matrix, rows, cols):
    return '\n'.join(' '.join(str(matrix[i][j].item()) for j in range(cols)) for i in range(rows))

def format_matrix_batch(matrix, rows, cols, batch_size):
    # ERROR HERE
    return '\n'.join(' '.join(str(matrix[k][i][j].item()) for j in range(cols)) for k in range(batch_size) for i in range(rows))

def prepare_string_V2(weights, dk, n_tokens, batch_size):

    # Prepare string with initial weights
    if batch_size is None:
        result = format_matrix(weights[0], n_tokens, dk) + '\n'
    else:
        result = format_matrix_batch(weights[0], n_tokens, dk, batch_size) + '\n'

    # Format and append additional weights matrices
    for i in range(1,5):
        result += format_matrix(weights[i], dk, dk) + '\n'
    
    if len(weights) == 8:
        result += format_matrix(weights[4], dk, dk) + '\n'
        result += format_matrix(weights[5], dk, dk*4) + '\n'
        result += format_matrix(weights[6], dk*4, dk) + '\n'
    
    return result

def prepare_string_sm(weights, n_tokens, batch_size, n_heads):

    result = '\n'.join(' '.join(str(weights[k][n][i][j].item()) for j in range(n_tokens)) for k in range(batch_size) for n in range(n_heads) for i in range(n_tokens)) + '\n'
    
    return result

