import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Loss_CE(nn.Module):

    def __init__(self, temperature : float = 0.07):
        super().__init__()
        self.temperature = temperature

    def cosine_similarity(self, x : torch.Tensor, y : torch.Tensor):
        dot_product = torch.mm(x, y.t())
        magnitude_x = torch.norm(x, dim=1)
        magnitude_y = torch.norm(y, dim=1)
        cosine_similarities = dot_product / torch.ger(magnitude_x, magnitude_y)

        return cosine_similarities
    
    def nt_xent(self, cos_sims : torch.Tensor):
        """
        Compute the normalized temperature scaled cross entropy loss.
        """
        loss = []

        for i in range(len(cos_sims)):
            neg_pairs = 0
            pos_pair = 0
            for j in range(len(cos_sims[i])):
                if i == j:
                    pos_pair = torch.exp(cos_sims[i][j] / self.temperature)
                else:
                    neg_pairs += torch.exp(cos_sims[i][j] / self.temperature)
            loss.append(-torch.log(pos_pair / neg_pairs))

        return torch.Tensor(loss)
    
    def total_loss(self, loss_e2t : torch.Tensor, loss_t2e : torch.Tensor):
        """
        Compute the total loss.
        """
        loss_sum = loss_e2t.sum() + loss_t2e.sum()

        loss = 1 / (2 * len(loss_e2t)) * loss_sum

        return loss

    def forward(self, ecg : torch.Tensor, text : torch.Tensor):

        # Compute the cosine similarities between the two tensors
        r_e2t = self.cosine_similarity(ecg, text)
        r_t2e = self.cosine_similarity(text, ecg)

        # Compute the normalized temperature scaled cross entropy loss
        loss_e2t = self.nt_xent(r_e2t)
        loss_t2e = self.nt_xent(r_t2e)

        loss = self.total_loss(loss_e2t, loss_t2e)

        loss.requires_grad = True

        return loss

class NT_XENT(nn.Module):
    def __init__(self):
        super().__init__()