from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
from models.linear_projectors import TextLinearProjectionHead

# Disable cache warning
HF_HUB_DISABLE_SYMLINKS_WARNING = True

class bio_clinical_BERT(nn.Module):
    """
    This class implements the BioClinicalBERT model.
    https://arxiv.org/pdf/1904.03323.pdf

    """
    def __init__(self, model_path : str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.name = "BioClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.linear_proj = TextLinearProjectionHead(768)
    
    def encode(self, text : list, add_special_tokens : bool):
        """
        Returns the encodings of the text input.
        """

        encoded_output = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", 
                                        return_attention_mask=True, return_token_type_ids=True,
                                        max_length=512, add_special_tokens=add_special_tokens)

        return encoded_output
    
    def embed(self, encoded_output):
        """
        Returns the embeddings of the encoded output.
        """
        output = self.model(**encoded_output)
        embeddings = output.last_hidden_state.mean(dim=1)

        return embeddings
    
    def forward(self, text : list, add_special_tokens : bool = False):
            """
            Returns the embeddings of the text input.

            Args:
                text (list): The input text to be encoded.
                add_special_tokens (bool, optional): Whether to add special tokens to the input. Defaults to False.

            Returns:
                embeddings: The embeddings of the input text.
            """

            # encoded_output = self.encode(text, add_special_tokens)
            with torch.no_grad():
                encoded_output = self.encode(text, add_special_tokens)
                embeddings = self.embed(encoded_output)
                text_proj = self.linear_proj(embeddings)
            return text_proj