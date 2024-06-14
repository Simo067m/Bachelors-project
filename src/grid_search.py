import torch
import torch.nn.functional as F
import wandb
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from models.resnet import ResNet, ResidualBlock
from models.linear_classifier import LinearClassifier
from dataloader.ptb_xl import ptb_xl_data_generator
from configs.ptb_xl_configs import Configs, TestingConfigs

def train_linear_classifier(ecg_model, classifier, train_loader, val_loader, num_epochs, optimizer, criterion, device):
    """
    Trains the linear classifier using the embeddings from the ECG and text models.
    Since both the ECG and text models are pre-trained, only the weights of the classifier are trained.
    """
    losses = []
    val_losses = []

    classifier.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
            
        for i, data in enumerate(train_loader):
            ecg, target = data

            ecg = ecg.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                ecg_output = ecg_model(ecg)

            output = classifier(ecg_output)

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
                ecg, target = data

                ecg = ecg.to(device)
                target = target.to(device)

                ecg_output = ecg_model(ecg)

                output = classifier(ecg_output)

                val_loss = criterion(output, target)
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        wandb.log({
            "Train loss": avg_loss,
            "Validation loss": avg_val_loss
        }, step=epoch + 1)

        classifier.train()
        
    return losses, val_losses
    
def test_linear_classifier_new(ecg_model, classifier, test_loader, device):
    """
    Tests the linear classifier.
    """
    classifier.eval()
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            ecg, target = data
            ecg = ecg.to(device)
            target = target.to(device)

            ecg_output = ecg_model(ecg)

            output = classifier(ecg_output)
            prob = F.softmax(output, dim=1)

            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(prob.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    # Calculate AUC score for each class
    y_true_one_hot = F.one_hot(torch.tensor(y_true), num_classes=output.size(1)).numpy()
    auc = roc_auc_score(y_true_one_hot, y_probs, average="macro", multi_class="ovr")

    accuracy = accuracy * 100
    f1 = f1 * 100
    auc = auc * 100

    return accuracy, f1, auc

def train():
    with wandb.init(config=parameters_dict):
        
        config = wandb.config

        configs = Configs(config.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 18, ResidualBlock).to(device)
        ecg_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "saved_models/ResNet18_20_epochs")))
        ecg_model.eval()
        classifier = LinearClassifier(configs).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr)
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=config.lr)
        train_loader, val_loader, test_loader = ptb_xl_data_generator(configs)
        losses, val_losses = train_linear_classifier(ecg_model, classifier, train_loader, val_loader, config.epochs, optimizer, criterion, device)
        accuracy, f1, auc = test_linear_classifier_new(ecg_model, classifier, test_loader, device)

        wandb.log({
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc,
        })

sweep_config = {
    "method": "grid"
}

parameters_dict = {
    "optimizer": {
        "values": ["adam", "sgd"]
    },
    "lr": {
        "values": [0.001, 0.01, 0.1, 0.05, 0.005]
    },
    "batch_size": {
        "values": [32, 64, 96, 128]
    },
    "epochs": {
        "values": [10, 20, 30, 40, 50]
    }
}

sweep_config["parameters"] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="Bachelors-project")

wandb.agent(sweep_id, function=train, count=50)