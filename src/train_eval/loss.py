import torch

def cosine_similarity(x : torch.Tensor, y : torch.Tensor):
    """
    Compute the cosine similarity between two tensors.
    Uses the formula: x.y / (||x|| * ||y||)
    """
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

def cosine_similarity_torch(x: torch.Tensor, y: torch.Tensor):
    """
    Compute the cosine similarity between two tensors.
    Uses the torch.nn.functional.cosine_similarity function.
    """
    return torch.nn.functional.cosine_similarity(x, y, dim = 0)