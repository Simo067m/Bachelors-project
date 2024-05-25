import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NT_Xent_loss(nn.Module):
    def __init__(self, batch_size, temperature=0.07):
        super(NT_Xent_loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity_function = nn.CosineSimilarity(dim=2)

    def forward(self, ecg_embeddings, text_embeddings):

        # Calculate similarity matrix with unsqueezing for every pair of similarities
        similarity_matrix_e2t = self.similarity_function(ecg_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0))
        similarity_matrix_t2e = self.similarity_function(text_embeddings.unsqueeze(1), ecg_embeddings.unsqueeze(0))

        # Compute the loss for ecg_embeddings -> text_embeddings
        loss_e2t = self.compute_loss(similarity_matrix_e2t)

        # Compute the loss for text_embeddings -> ecg_embeddings
        loss_t2e = self.compute_loss(similarity_matrix_t2e)

        # Get the total loss
        total_loss = (loss_e2t + loss_t2e) / 2 # Do not divide by batch_size here as the loss was already averaged in the compute_loss function by F.cross_entropy

        return total_loss

    def compute_loss(self, similarity_matrix):
        # Get the similarities of the positive pairs
        positive_sims = torch.diag(similarity_matrix)

        # Create a mask to retrieve all the similarities of the negative pairs. The mask is of type bool and is used to exclude positive pair similarities.
        negatives_mask = torch.eye(self.batch_size, device=similarity_matrix.device, dtype=torch.bool)

        # Invert the mask so only the diagonal elements are false and all other values become true
        negatives_mask = ~negatives_mask

        # Get the negatives using the mask
        negative_sims = similarity_matrix[negatives_mask] # tensor
        negative_sims = similarity_matrix[negatives_mask].view(self.batch_size, -1) # matrix

        # Apply the temperature scaling
        positive_sims /= self.temperature
        negative_sims /= self.temperature

        # Concatenate the positive and negative similarities to form the logits
        logits = torch.cat((positive_sims.unsqueeze(1), negative_sims), dim=1)
        
        # Compute the labels for the loss. Using torch.nn.functional.cross_entropy with a labels tensor of all zeros is equivalent to the first column of the logits tensor being the target.
        labels = torch.zeros(self.batch_size, device=similarity_matrix.device, dtype=torch.long)

        # Computing the loss manually can be done as below, however, it is not recommended due to numerical stability issues.
        """losses = []
        for i in range(self.batch_size):
            losses.append(-torch.log(torch.exp(logits[i, 0]) / torch.exp(logits[i]).sum()))
        man_loss = torch.tensor(losses).sum() """

        # Compute the loss using torch.nn.functional.cross_entropy
        loss = F.cross_entropy(logits, labels)

        return loss