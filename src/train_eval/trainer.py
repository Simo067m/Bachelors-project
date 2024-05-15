"""
This script contains the Trainer class which is used to train and evaluate the model.
"""

import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torcheval.metrics.functional import multiclass_f1_score

class Trainer:
    """
    Class to train and evaluate torch.nn models.
    """
    def __init__(self, log_wandb : bool = False, wandb_configs : object = None, save_model : bool = False):
        self.log_wandb = log_wandb
        self.save_model = save_model
        if self.log_wandb:
            wandb.init(
                project="Feature-Testing",

                config={
                    "learning_rate" : 0.002,
                    "weight_decay" : 0.00001,
                    "architecture" : "ResNet-18",
                    "dataset" : "PTB-XL",
                    "epochs" : 100,
                }
            )
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

    def ecg_encoder_pre_train(self, ecg_model, text_model, train_loader, num_epochs, optimizer, criterion, device, save_path = None):
        """
        Pre-trains the ecg encoder.
        Only the ecg model weights are trained, as the pre-trained text model weights are frozen.
        """
        losses = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            
            for i, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1} / {num_epochs}", total=len(train_loader)):
                ecg, text, target = data
                ecg = ecg.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                ecg_output = ecg_model(ecg)
                text_output = text_model(text).to(device) # Text was on cpu but needs to be on the same device as the ecg model

                loss = criterion(ecg_output, text_output)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            losses.append(running_loss / len(train_loader))
        
        print(f"Finished pre-training {ecg_model.name}.")

        if save_path is not None:
            torch.save(ecg_model.state_dict(), save_path)
            print(f"Model saved at {save_path}.")
        
        return losses

