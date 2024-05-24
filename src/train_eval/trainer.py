"""
This script contains the Trainer class which is used to train and evaluate the model.
"""

import torch
from tqdm import tqdm
import numpy as np
import wandb
import os
import time
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torcheval.metrics.functional import multiclass_f1_score

class Trainer:
    """
    Class to train and evaluate torch.nn models.
    """
    def __init__(self):
        pass

    def ecg_encoder_pre_train(self, ecg_model, text_model, train_loader, val_loader, num_epochs, optimizer, criterion, device, save_name = None):
        """
        Pre-trains the ecg encoder.
        Only the ecg model weights are trained, as the pre-trained text model weights are frozen.
        """
        start_time = time.time()
        losses = []
        val_losses = []
        ecg_model.train()
        text_model.eval() # Text model is frozen
        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for i, data in enumerate(train_loader):
                ecg, text, target = data

                ecg = ecg.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                ecg_output = ecg_model(ecg)

                with torch.no_grad():  # Ensure the text model does not backpropagate gradients
                    text_output = text_model(text).to(device)  # Get text embeddings

                loss = criterion(ecg_output, text_output)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)

            # Validation step
            ecg_model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    ecg, text, target = data

                    ecg = ecg.to(device)
                    target = target.to(device)

                    ecg_output = ecg_model(ecg)
                    text_output = text_model(text).to(device)  # Get text embeddings

                    val_loss = criterion(ecg_output, text_output)
                    val_running_loss += val_loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if save_name is not None:
                wandb.log({"Train Loss" : avg_loss, "Val Loss" : avg_val_loss}, step=epoch + 1)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            
            # Print elapsed time
            print(f"Elapsed time: {hours} hours and {minutes} minutes.")

            ecg_model.train()
        
        print(f"Finished pre-training {ecg_model.name}.")
        print(f"Elapsed time: {hours} hours and {minutes} minutes.")

        if save_name is not None:
            save_path = os.path.join(os.getcwd(), "saved_models", save_name)
            torch.save(ecg_model.state_dict(), save_path)
            print(f"Model saved at {save_path}.")
        
        return losses, val_losses

    # From chat_gpt
    def evaluate_ecg_encoder(self, ecg_model, text_model, test_loader, device):
        """
        Evaluates the ECG encoder by comparing its embeddings with text embeddings.
        """
        ecg_model.eval()
        text_model.eval()
        
        similarities = []
        correct = 0
        total = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader), desc="Evaluating", total=len(test_loader)):
                ecg, text, target = data
                ecg = ecg.to(device)
                
                ecg_output = ecg_model(ecg)
                text_output = text_model(text).to(device)

                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(ecg_output, text_output, dim=-1)
                similarities.append(similarity.cpu().numpy())

                # Evaluate using a simple threshold
                preds = (similarity > 0.5).float()  # Example threshold for positive pair classification
                correct += (preds == target.to(device)).sum().item()
                total += target.size(0)
        
        avg_similarity = np.mean(np.concatenate(similarities))
        accuracy = correct / total
        
        print(f"Average Similarity: {avg_similarity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return avg_similarity, accuracy
    
    def train_linear_classifier(self, ecg_model, text_model, classifier, train_loader, val_loader, num_epochs, optimizer, criterion, device, save_name = None):
        """
        Trains the linear classifier using the embeddings from the ECG and text models.
        Since both the ECG and text models are pre-trained, only the weights of the classifier are trained.
        """
        start_time = time.time()
        losses = []
        val_losses = []

        classifier.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for i, data in enumerate(train_loader):
                ecg, text, target = data

                ecg = ecg.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                with torch.no_grad():
                    ecg_output = ecg_model(ecg)
                    text_output = text_model(text).to(device)

                output = classifier(ecg_output, text_output)

                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)

            # Validation step
            classifier.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    ecg, text, target = data

                    ecg = ecg.to(device)
                    target = target.to(device)

                    ecg_output = ecg_model(ecg)
                    text_output = text_model(text).to(device)

                    output = classifier(ecg_output, text_output)

                    val_loss = criterion(output, target)
                    val_running_loss += val_loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if save_name is not None:
                wandb.log({"Train Loss" : avg_loss, "Val Loss" : avg_val_loss}, step=epoch + 1)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            
            # Print elapsed time
            print(f"Elapsed time: {hours} hours and {minutes} minutes.")

            classifier.train()

        print(f"Finished training {classifier.name}.")
        print(f"Elapsed time: {hours} hours and {minutes} minutes.")

        if save_name is not None:
            save_path = os.path.join(os.getcwd(), "saved_models", save_name)
            torch.save(ecg_model.state_dict(), save_path)
            print(f"Model saved at {save_path}.")
        
        return losses, val_losses
        
    def test_linear_classifier(self, ecg_model, text_model, classifier, test_loader, device):
        """
        Tests the linear classifier.
        """
        classifier.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
                ecg, text, target = data
                ecg = ecg.to(device)
                target = target.to(device)

                ecg_output = ecg_model(ecg)
                text_output = text_model(text).to(device)

                output = classifier(ecg_output, text_output)

                _, predicted = torch.max(output.data, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, f1