from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import time

# Disable cache warning
HF_HUB_DISABLE_SYMLINKS_WARNING = True

class bio_clinical_BERT(nn.Module):
    """
    This class implements the BioClinicalBERT model.
    https://arxiv.org/pdf/1904.03323.pdf

    """
    def __init__(self, model_path : str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
    
    def encode(self, text : list, add_special_tokens : bool):
        """
        Returns the encodings of the text input.
        """
        start_time = time.time()

        encoded_output = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", 
                                        return_attention_mask=True, return_token_type_ids=True,
                                        max_length=512, add_special_tokens=add_special_tokens)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Finished encoding text in {minutes} minutes and {seconds} seconds.")

        return encoded_output
    
    def embed(self, encoded_output):
        """
        Returns the embeddings of the encoded output.
        """

        start_time = time.time()

        with torch.no_grad():
            output = self.model(**encoded_output)
            embeddings = output.last_hidden_state.mean(dim=1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Finished embedding text in {minutes} minutes and {seconds} seconds.")

        return embeddings
    
    def forward(self, text : list, add_special_tokens : bool = False):
        """
        Returns the embeddings of the text input.
        """
        encoded_output = self.encode(text, add_special_tokens)
        embeddings = self.embed(encoded_output)
        return embeddings