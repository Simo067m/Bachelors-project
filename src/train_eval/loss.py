import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
class NTXent_loss(nn.Module):
    def __init__(self, batch_size, temperature=0.07, device="cuda"):
        """Compute the NT-Xent loss for contrastive learning.
        k = batch_size
        """
        super(NTXent_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.negatives_mask = self.get_correlated_mask().to(self.device)
        self.similarity_function = nn.CosineSimilarity(dim=2)
        self.criterion  = nn.CrossEntropyLoss(reduction="sum")
    
    def get_correlated_mask(self):
        """Generate a mask to prevent positive pairs from being selected"""
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k = -self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k = self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)
    
    def _get_correlated_mask_with_anti_diagonal(self):
        mask = torch.ones((self.batch_size, self.batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size - i - 1] = 0
        return mask
    
    def compute_loss(self, similarity_matrix):

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.negatives_mask].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)

        targets = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, targets)

        return loss
    
    def forward(self, x, y):
        representations = torch.cat([x, y], dim=0)
        similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0)) / self.temperature

        representations_rev = torch.cat([y, x], dim=0)
        similarity_matrix_rev = self.similarity_function(representations_rev.unsqueeze(1), representations_rev.unsqueeze(0)) / self.temperature

        loss = self.compute_loss(similarity_matrix)
        loss_rev = self.compute_loss(similarity_matrix_rev)

        total_loss = (loss + loss_rev) / (2 * self.batch_size)

        return total_loss
    
class NTXent_loss_new(nn.Module):
    def __init__(self, batch_size):
        super(NTXent_loss_new, self).__init__()
        self.batch_size = batch_size

    def forward(self, tensor1, tensor2):
        # Compute cosine similarity for each pair in the batch
        similarity_matrix = F.cosine_similarity(tensor1.unsqueeze(1), tensor2.unsqueeze(0), dim=2)

        # Exclude diagonal elements (similarity between tensors with the same index)
        mask = torch.eye(self.batch_size, dtype=torch.bool)
        non_diag_similarities = similarity_matrix[~mask].view(self.batch_size, self.batch_size - 1)

        # Positive similarities: diagonal elements
        pos_sim = similarity_matrix.diag()

        # Compute loss in a numerically stable manner
        temperature = 0.07

        # Numerator for each positive pair
        numerator = torch.exp(pos_sim / temperature)

        # Denominator: logsumexp for each positive pair's negative similarities
        denominator = torch.logsumexp(non_diag_similarities / temperature, dim=1)

        # Compute the loss
        loss = -torch.mean(numerator - denominator)

        return loss