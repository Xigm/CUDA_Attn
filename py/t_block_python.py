import torch

def simple_attention_block(inputs, W_q, W_k, W_v, W_cproj, W_li, W_lo, dk):

    # Define the input tensors
    query = torch.matmul(inputs, W_q)
    key = torch.matmul(inputs, W_k)
    value = torch.matmul(inputs, W_v)

    # apply layer norm
    inputs_ln = torch.nn.functional.LayerNorm(inputs)

    # Compute the attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())

    # apply casual mask
    attention_scores = attention_scores.masked_fill(torch.tril(torch.ones((inputs_ln.shape[0],inputs_ln.shape[0]))) == 0, float('-inf'))

    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Apply attention weights to the value tensor
    output = torch.matmul(attention_weights, value)

    # Apply attention output projection
    output = torch.matmul(output, W_cproj)

    # apply residual connection
    output += inputs

    # apply layer norm
    output_ln = torch.nn.LayerNorm(output)

    # Apply the linear transformation
    output_ln = torch.matmul(output_ln, W_li)
    output_ln = torch.matmul(output_ln, W_lo)

    # apply residual connection
    output += output_ln

    # Print the result
    return output


