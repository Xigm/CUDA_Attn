import torch
import time
def simple_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk):

    # Define the input tensors
    query = torch.matmul(inputs, W_q)
    key = torch.matmul(inputs, W_k)
    value = torch.matmul(inputs, W_v)

    # print(query)

    # Compute the attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    # print(attention_scores)

    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())

    # print(attention_scores)

    # apply casual mask
    if len(inputs.shape) == 3:
        attention_scores = attention_scores.masked_fill(torch.tril(torch.ones((inputs.shape[0],inputs.shape[1],inputs.shape[1]))) == 0, float('-inf'))
    else:
        attention_scores = attention_scores.masked_fill(torch.tril(torch.ones((inputs.shape[0],inputs.shape[0]))) == 0, float('-inf'))
        
    # print(attention_scores)

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # print(attention_weights)

    # Apply attention weights to the value tensor
    output = torch.matmul(attention_weights, value)

    # print(output)

    # Apply attention output projection
    output = torch.matmul(output, W_cproj)

    # print(output)

    # Print the result
    return output


def gpu_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk, mask):

    # Define the input tensors
    query = torch.matmul(inputs, W_q)
    key = torch.matmul(inputs, W_k)
    value = torch.matmul(inputs, W_v)

    # print(query)

    # Compute the attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())

    # apply casual mask
    attention_scores = attention_scores.masked_fill(mask, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # print(attention_weights)

    # Apply attention weights to the value tensor
    output = torch.matmul(attention_weights, value)

    # print(output)

    # Apply attention output projection
    output = torch.matmul(output, W_cproj)

    # print(output)

    # Print the result
    return output


def gpu_self_attention_timed(inputs, W_q, W_k, W_v, W_cproj, dk, mask):
    """ version of the function that returns the time taken to perform each operation"""

    # Define the input tensors
    time_start = time.time()
    query = torch.matmul(inputs, W_q)
    key = torch.matmul(inputs, W_k)
    value = torch.matmul(inputs, W_v)
    time_query_key_value = time.time() - time_start

    # print(query)

    # Compute the attention scores
    time_start = time.time()
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    time_attention_scores = time.time() - time_start

    time_start = time.time()
    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())
    time_attention_scores_divide = time.time() - time_start

    # apply casual mask
    time_start = time.time()
    attention_scores = attention_scores.masked_fill(mask, float('-inf'))
    time_attention_scores_mask = time.time() - time_start

    # Apply softmax to get attention weights
    time_start = time.time()
    attention_weights = torch.softmax(attention_scores, dim=-1)
    time_attention_weights = time.time() - time_start

    # print(attention_weights)

    # Apply attention weights to the value tensor
    time_start = time.time()
    output = torch.matmul(attention_weights, value)
    time_output = time.time() - time_start

    # print(output)

    # Apply attention output projection
    time_start = time.time()
    output = torch.matmul(output, W_cproj)
    time_output_projection = time.time() - time_start

    # print(output)

    # print all times
    print("Time taken to compute query, key, and value matrices: ", time_query_key_value)
    print("Time taken to compute attention scores: ", time_attention_scores)
    print("Time taken to divide attention scores: ", time_attention_scores_divide)
    print("Time taken to mask attention scores: ", time_attention_scores_mask)
    print("Time taken to compute attention weights: ", time_attention_weights)
    print("Time taken to compute output: ", time_output)
    print("Time taken to compute output projection: ", time_output_projection)


    # Print the result
    return output




