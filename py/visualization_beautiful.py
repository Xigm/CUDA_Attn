
import torch

def MH_self_attention(inputs, W_q, W_k, W_v, W_cproj, dk, n_tokens,  n_heads, batch_size, mask = None):

    query = torch.matmul(inputs, W_q).view(batch_size, n_tokens, n_heads, dk//n_heads).transpose(1, 2)
    key = torch.matmul(inputs, W_k).view(batch_size, n_tokens, n_heads, dk//n_heads).transpose(1, 2)
    value = torch.matmul(inputs, W_v).view(batch_size, n_tokens, n_heads, dk//n_heads).transpose(1, 2)

    attention_scores = torch.matmul(query, key.transpose(-2, -1))

    attention_scores = attention_scores / torch.sqrt(torch.tensor(dk).float())

    attention_scores = attention_scores.masked_fill(torch.tril(torch.ones(attention_scores.shape)) == 0, float('-inf'))  
    
    attention_weights = torch.softmax(attention_scores, dim=-1)

    output = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, n_tokens, dk)

    output = torch.matmul(output, W_cproj)

    return output






