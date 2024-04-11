import torch

# define the input tensors as tensors of length 3
inputs = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float32)
print(inputs.shape)

# define the query, key, and value projection matrices
W_q = torch.tensor([[1,0],[0,1],[1,1]], dtype=torch.float32)
W_k = torch.tensor([[1,1],[0,1],[1,0]], dtype=torch.float32)
W_v = torch.tensor([[0,1],[1,1],[1,0]], dtype=torch.float32)

# Define the input tensors
query = torch.matmul(inputs, W_q)
key = torch.matmul(inputs, W_k)
value = torch.matmul(inputs, W_v)

# Compute the attention scores
attention_scores = torch.matmul(query, key.transpose(-2, -1))
attention_scores = attention_scores / torch.sqrt(torch.tensor(query.size(-1)).float())

# Apply softmax to get attention weights
attention_weights = torch.softmax(attention_scores, dim=-1)

# Apply attention weights to the value tensor
output = torch.matmul(attention_weights, value)

# Print the result
print(output)