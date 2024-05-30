import torch
import torch.nn.functional as F

class ZeroShotClassifier():
    """
    Class for performing Zero-shot classification.

    Functions take unseen text prompts and ECG samples and classifies based on the largest cosine similarity.

    Prompts need to include "{}" brackets which will be filled in with the relevant text value based on the task.
    """
    def __init__(self):
        self.label_map = {0 : "CD", 1 : "HYP", 2 : "MI", 3 : "NORM", 4 : "STTC"}
    
    def fill_sentence(self, prompt : str, word_to_fill : str):
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

    def single_class_classification(self, ecg_model, text_model, train_loader, positive_prompt, negative_prompt, device):

        # Generate text embeddings for every possible propmt after filling in class values to save computing time
        possible_positive_prompts = [self.fill_sentence(positive_prompt, value) for value in self.label_map.values()]
        possible_negative_prompts = [self.fill_sentence(negative_prompt, value) for value in self.label_map.values()]
        positive_embeddings = {prompt : text_model(prompt).to(device) for prompt in possible_positive_prompts}
        negative_embeddings = {prompt : text_model(prompt).to(device) for prompt in possible_negative_prompts}

        for i, data in enumerate(train_loader):
            ecg, targets = data

            ecg = ecg.to(device)

            # Get the ecg embeddings
            with torch.no_grad():
                ecg_embeddings = ecg_model(ecg)

            # Get the target labels
            target_labels = [self.label_map[target.item()] for target in targets]
            
            # Fill in the positive and negative prompts with the target labels
            positive_prompts = [self.fill_sentence(positive_prompt, target_label) for target_label in target_labels]
            negative_prompts = [self.fill_sentence(negative_prompt, target_label) for target_label in target_labels]

            # Make pairs of an ECG sample, and the corresponding positive and negative text prompt embeddings
            for i in range(len(target_labels)):
                pairs = {"target_label" : target_labels[i], "ecg_sample" : ecg_embeddings[i], "positive_embedding" : positive_embeddings[positive_prompts[i]], "negative_embedding" : negative_embeddings[negative_prompts[i]]}
            
            # Perform the zero-shot classification
            for i, target_label in enumerate(target_labels):
                ecg_sample = ecg_embeddings[i]
                positive_embedding = positive_embeddings[positive_prompts[i]]
                negative_embedding = negative_embeddings[negative_prompts[i]]

                # Calculate the cosine similarity between the ECG sample and the positive and negative text embeddings
                positive_similarity = F.cosine_similarity(ecg_sample, positive_embedding)
                negative_similarity = F.cosine_similarity(ecg_sample, negative_embedding)

                # Determine the predicted label based on the similarity scores
                if positive_similarity > negative_similarity:
                    predicted_label = "Positive"
                else:
                    predicted_label = "Negative"
                
                print(predicted_label, target_label)

                # TODO: This right now searches for the label every time. Instead, change so that one class is picked for an entire run, so EVERY prompt is fx. "NORM" or "not NORM", and then predict whether or not the ecg sample actually is. Right now this does essentially nothing.

            break
