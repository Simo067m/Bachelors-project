"""
Script containing functions for data processing.
Functions in this script are meant for encoding data.
"""

import torch

def process_text(text_model, text_data):
    """
    This function processes the text data using the text model.
    """
    # Encode the text data
    encoded_output = text_model.encode(text_data, add_special_tokens=True)
    
    # Embed the encoded output
    embeddings = text_model.embed(encoded_output)
    
    return embeddings

def process_ecg(ecg_model, ecg_data):
    """
    This function processes the ECG data using the ECG model.
    """
    with torch.no_grad():
        output = ecg_model(ecg_data)
    
    return output