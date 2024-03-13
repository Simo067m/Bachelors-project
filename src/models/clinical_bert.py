from transformers import AutoTokenizer, AutoModel
import torch

# Disable cache warning
HF_HUB_DISABLE_SYMLINKS_WARNING = True

# Load the model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

sentence = "The patient is in danger of suffering a heart attack."

encoded_input = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

print(encoded_input)

with torch.no_grad():
    output = model(**encoded_input)
    embeddings = output.last_hidden_state.mean(dim=1)

print(embeddings)
print(len(embeddings[0]))
print(len(sentence))

class bio_clinical_BERT:
    """
    This class implements the BioClinicalBERT model.
    https://arxiv.org/pdf/1904.03323.pdf

    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")