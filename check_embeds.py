import torch

# Load the saved embeddings (using weights_only=True is a good security practice)
null_dict = torch.load("null_embeddings.pt", weights_only=True)

print("--- Null Embeddings Inspection ---")
for key, tensor in null_dict.items():
    print(f"Key: {key:<15} | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")