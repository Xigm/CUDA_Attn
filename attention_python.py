import torch

def simple_self_attention(inputs, W_q, W_k, W_v, dk):

    # Define the input tensors
    query = torch.matmul(inputs, W_q)
    key = torch.matmul(inputs, W_k)
    value = torch.matmul(inputs, W_v)

    # Compute the attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Apply attention weights to the value tensor
    output = torch.matmul(attention_weights, value)

    # Print the result
    return output


