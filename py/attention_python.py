import torch

def simple_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk):

    # Define the input tensors
    query = torch.matmul(inputs, W_q)
    key = torch.matmul(inputs, W_k)
    value = torch.matmul(inputs, W_v)

    print(query[:3, :3])

    # Compute the attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())

    # apply casual mask
    attention_scores = attention_scores.masked_fill(torch.tril(torch.ones((inputs.shape[0],inputs.shape[0]))) == 0, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)
    # print(attention_weights)

    # Apply attention weights to the value tensor
    output = torch.matmul(attention_weights, value)

    # Apply attention output projection
    output = torch.matmul(output, W_cproj)

    # Print the result
    return output


