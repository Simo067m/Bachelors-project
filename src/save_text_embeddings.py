from models.clinical_bert import bio_clinical_BERT
from configs.ptb_xl_configs import Configs
import pandas as pd
import torch

def fill_sentence(prompt : str, word_to_fill : str):
        """
        Replaces the placeholders in the template string with the provided words.
        
        Parameters:
            template_string (str): The template string with placeholders.
            word (list): A list of words to fill in the placeholders.
            
        Returns:
            str: The filled-in string.
        """
        filled_string = prompt.format(word_to_fill)
    
        return filled_string

configs = Configs(128)

text_data = pd.read_csv(configs.path_to_dataset+"saved_splits/data_translated_unconfirmed_removed.csv")["translated_report"].tolist()
text_data_prompt = [fill_sentence("The report of the ECG is that {}", text) for text in text_data]

model = bio_clinical_BERT()

text_embeddings = {}
text_embeddings_prompt = {}

for text in text_data:
    if text not in text_embeddings:
        embedding = model(text)
        text_embeddings[text] = embedding

for text in text_data_prompt:
    if text not in text_embeddings_prompt:
        embedding = model(text)
        text_embeddings_prompt[text] = embedding

torch.save(text_embeddings, configs.path_to_dataset+"saved_splits/text_embeddings.pt")
torch.save(text_embeddings_prompt, configs.path_to_dataset+"saved_splits/text_embeddings_prompt.pt")