import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

class ZeroShotClassifier():
    """
    Class for performing Zero-shot classification.

    Functions take unseen text prompts and ECG samples and classifies based on the largest cosine similarity.

    Prompts need to include "{}" brackets which will be filled in with the relevant text value based on the task.
    """
    def __init__(self, dataset : str):
        if dataset == "ptb-xl":
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

        ecg, targets = train_loader.dataset[:]

        ecg = ecg.to(device)
        
        # Get the target labels
        target_labels = [self.label_map[target.item()] for target in targets]

        # Get the ecg embeddings
        with torch.no_grad():
            ecg_embeddings = ecg_model(ecg)

        results = {}
        
        # Perform the zero-shot classification. Performed once for every possible class
        for target_class in self.label_map.values():
            correct = 0
            y_true = []
            y_pred = []
            y_scores = []
            positive_embedding = positive_embeddings[self.fill_sentence(positive_prompt, target_class)]
            negative_embedding = negative_embeddings[self.fill_sentence(negative_prompt, target_class)]
            
            for i, label in enumerate(target_labels):
                ecg_sample = ecg_embeddings[i]
                positive_similarity = F.cosine_similarity(ecg_sample, positive_embedding)
                negative_similarity = F.cosine_similarity(ecg_sample, negative_embedding)

                predicted_label = positive_similarity > negative_similarity
                    
                # Check if the predicted label was correct
                """if predicted_label:
                    if label == target_class:
                        correct += 1
                else:
                    if label != target_class:
                        correct += 1"""
                if (predicted_label and label == target_class) or (not predicted_label and label != target_class):
                    correct += 1
                
                y_true.append(1 if label == target_class else 0)
                y_pred.append(1 if predicted_label else 0)
                y_scores.append(positive_similarity.item())
            
            accuracy = (correct / len(target_labels)) * 100
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else float('nan')

            results[target_class] = {
                "accuracy": accuracy,
                "f1_score": f1 * 100,
                "auc_score": auc * 100
            }
            print(f"Class {target_class} - Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, AUC Score: {auc:.4f}")
        
        results["average"] = {
            "accuracy": sum([result["accuracy"] for result in results.values()]) / len(results),
            "f1_score": sum([result["f1_score"] for result in results.values()]) / len(results),
            "auc_score": sum([result["auc_score"] for result in results.values()]) / len(results)
        }
        print(f"Average - Accuracy: {results['average']['accuracy']:.2f}%, F1 Score: {results['average']['f1_score']:.4f}, AUC Score: {results['average']['auc_score']:.4f}")
                    
        return results