import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def check_sims(batch_size : int, tensor1 : torch.Tensor, tensor2 : torch.Tensor):
    """
    Compute the average non-diagonal similarity and average diagonal similarity between two tensors.

    Args:
        configs (object): Configuration object containing batch size.
        tensor1 (torch.Tensor): First tensor.
        tensor2 (torch.Tensor): Second tensor.

    Returns:
        tuple: A tuple containing the average non-diagonal similarity and average diagonal similarity.

    """
    # Compute cosine similarity for each pair in the batch
    similarity_matrix = F.cosine_similarity(tensor1.unsqueeze(1), tensor2.unsqueeze(0), dim=2)

    # Exclude diagonal elements (similarity between tensors with the same index)
    mask = torch.eye(batch_size, dtype=torch.bool)
    non_diag_similarities = similarity_matrix[~mask].view(batch_size, -1)

    # Compute average similarity for non-diagonal elements
    avg_negative_similarity = non_diag_similarities.mean()
    avg_positive_similarity = similarity_matrix.diag().mean()

    #print(f'Average Non-Diagonal Similarity: {avg_non_diag_similarity.item()}')
    #print(f'Average Diagonal Similarity: {avg_diag_similarity.item()}')

    return avg_negative_similarity.item(), avg_positive_similarity.item()

def plot_losses(losses, val_losses):
    pass