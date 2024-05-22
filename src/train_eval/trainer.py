"""
This script contains the Trainer class which is used to train and evaluate the model.
"""

import torch
from tqdm import tqdm
import numpy as np
import wandb
import os
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torcheval.metrics.functional import multiclass_f1_score

class Trainer:
    """
    Class to train and evaluate torch.nn models.
    """
    def __init__(self):
        pass
    def train(self, model, train_loader, val_loader, num_epochs, optimizer, criterion, device, save_path = None):
        """
        Trains the model using the provided training data.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): The data loader for the training data.
            epochs (int): The number of epochs to train the model.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            criterion (torch.nn.Module): The loss function used for training.
            device (str): The device to be used for training (e.g., 'cpu', 'cuda').

        Returns:
            None
        """

        # Train the model

        total_loss = []
        for epoch in range(num_epochs):
            # Set model to train
            model.train()
            running_loss = 0.0

            for i, data in tqdm(enumerate(train_loader, 0), desc=f'Epoch {epoch + 1}/{num_epochs}', total=len(train_loader)):
                # Get the inputs and labels
                inputs, labels = data

                # Move the inputs and labels to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
            
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    _, predicted = torch.max(model(inputs), 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                
            print(f"Epoch {epoch + 1} Loss: {running_loss/77:.4f}")
            print(f"Validation accuracy: {total_correct/total_samples:.2f}")
            if self.log_wandb:
                wandb.log({"Loss" : running_loss/77, "Validation Accuracy" : total_correct/total_samples})
            
            total_loss.append(running_loss)
        print("Finished Training")
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
        return total_loss
    
    def test(self, model, test_loader, device):
        """
        Tests the model using the provided test data.
        """
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs), 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        print(f"Test accuracy: {total_correct/total_samples:.2f}")

        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, 1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = multiclass_f1_score(torch.tensor(all_predictions), torch.tensor(all_labels), num_classes=5, average="macro")

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")

    def ecg_encoder_pre_train(self, ecg_model, text_model, train_loader, val_loader, num_epochs, optimizer, criterion, device, save_name = None):
        """
        Pre-trains the ecg encoder.
        Only the ecg model weights are trained, as the pre-trained text model weights are frozen.
        """
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
                for i, data in tqdm(enumerate(val_loader), desc=f"Validation {epoch + 1} / {num_epochs}", total=len(val_loader)):
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
                wandb.log({"Epoch" : epoch + 1, "Train Loss" : avg_loss, "Val Loss" : avg_val_loss})

            ecg_model.train()
        
        print(f"Finished pre-training {ecg_model.name}.")

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