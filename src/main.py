# Import packages
import argparse
import os
import wandb
import torch
from torch.utils.data import DataLoader

# Import custom modules
from dataloader.ptb_xl import ptb_xl_data_generator
from models.clinical_bert import bio_clinical_BERT
from models.resnet import ResNet, ResidualBlock
from train_eval.trainer import Trainer

# Remove irrelevant pytorch storage warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def parse_args():

    parser = argparse.ArgumentParser(description='Representation learning of multimodal data using multiview machine learning')
    
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("-ptb-xl", action="store_const", help="PTB-XL dataset", dest="dataset", const="ptb-xl")

    text_model = parser.add_mutually_exclusive_group()
    text_model.add_argument("-bioclinicalbert", action="store_const", help="BioClinicalBERT model", dest="text_model", const="bioclinicalbert")

    ecg_model = parser.add_mutually_exclusive_group()
    ecg_model.add_argument("-resnet18", action="store_const", help="ResNet18 model", dest="ecg_model", const="resnet18")
    ecg_model.add_argument("-resnet34", action="store_const", help="ResNet34 model", dest="ecg_model", const="resnet34")
    
    return parser.parse_args()

def validate_args(args):
    if not any([args.dataset, args.text_model, args.ecg_model]):
        raise ValueError("At least one option should be selected.")

if __name__ == "__main__":
    args = parse_args()

    # Set the device to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset

    if args.dataset == "ptb-xl":
        dataset_name = "PTB-XL"
        
        # Import configs
        from configs.ptb_xl_configs import Configs
        configs = Configs()

        # Define dataset variables TODO: Add this to the argparser
        train_loader, val_loader, test_loader = ptb_xl_data_generator(configs)
        
    
    # Load the text model
        
    if args.text_model == "bioclinicalbert":
        text_model_name = "BioClinicalBERT"
        
        # Define text model variables TODO: Add this to the argparser
        text_model = bio_clinical_BERT()

    # Load the ECG model
        
    if args.ecg_model == "resnet18":
        ecg_model_name = "ResNet-18"
        
        """
        Notes on training: Hyperparameters from ETP paper section 3.2
        Learning rate: 2 * e^-3: 0.002
        Weight decay: 1 * e^-5: 0.00001
        During pre-training: Epochs: 50, batch size: 128
        Downstream tasks: Batch size: 32
        """

        # Define ECG model variables TODO: Add this to the argparser
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 18, ResidualBlock).to(device)
    
    elif args.ecg_model == "resnet34":
        ecg_model_name = "ResNet-34"
        print("ResNet-34 ECG model selected.")
        
        # Define ECG model variables TODO: Add this to the argparser
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 34, ResidualBlock).to(device)
    
    # Print names of the selected models
    print(f"Selected dataset: {dataset_name}")
    print(f"Selected text model: {text_model_name}")
    print(f"Selected ECG model: {ecg_model_name}")

    # Define the optimizer and criterion TODO: Add this to the argparser
    optimizer = torch.optim.Adam(ecg_model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer()

    # Train the model
    if torch.cuda.is_available():
        ecg_model = ecg_model.cuda()
    
    trainer.train(ecg_model, train_loader, val_loader, 3, optimizer, criterion, device, "hpc_saved.pth")
    trainer.test(ecg_model, test_loader, device)

    print("Done!")