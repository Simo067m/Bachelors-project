# Import packages
import argparse
import os
import wandb
import torch

# Import custom modules
from dataloader.ptb_xl import ptb_xl_dataset
from models.clinical_bert import bio_clinical_BERT
from models.resnet import ResNet, ResidualBlock

# Remove irrelevant pytorch storage warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def parse_args():

    parser = argparse.ArgumentParser(description='Representation learning of multimodal data using multiview machine learning')
    
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("-ptb-xl", action="store_const", help="PTB-XL dataset", dest="dataset", const="ptb-xl")

    text_model = parser.add_mutually_exclusive_group()
    text_model.add_argument("-bio-clinical-bert", action="store_const", help="BioClinicalBERT model", dest="text_model", const="bio-clinical-bert")

    ecg_model = parser.add_mutually_exclusive_group()
    ecg_model.add_argument("-resnet18", action="store_const", help="ResNet18 model", dest="ecg_model", const="resnet18")
    ecg_model.add_argument("-resnet34", action="store_const", help="ResNet34 model", dest="ecg_model", const="resnet34")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set the device to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset

    if args.dataset == "ptb-xl":
        dataset_name = "PTB-XL"
        print("PTB-XL dataset selected.")
        
        # Import configs
        from configs.ptb_xl_configs import Configs
        configs = Configs()

        # Define dataset variables TODO: Add this to the argparser
        dataset = ptb_xl_dataset(path_to_dataset = configs.path_to_dataset, sampling_rate = configs.sampling_rate, test_fold = configs.test_fold)
        # Permute the tensor dataset so that the channels are the first dimension
        dataset.X_train_ecg_tensor = dataset.X_train_ecg_tensor.permute(0, 2, 1)
        dataset.X_test_ecg_tensor = dataset.X_test_ecg_tensor.permute(0, 2, 1)
    
    # Load the text model
        
    if args.text_model == "bio-clinical-bert":
        text_model_name = "BioClinicalBERT"
        print("BioClinicalBERT text model selected.")
        
        # Define text model variables TODO: Add this to the argparser
        text_model = bio_clinical_BERT()

    # Load the ECG model
        
    if args.ecg_model == "resnet18":
        ecg_model_name = "ResNet-18"
        print("ResNet-18 ECG model selected.")
        
        """
        Notes on training: Hyperparameters from ETP paper section 3.2
        Learning rate: 2 * e^-3: 0.002
        Weight decay: 1 * e^-5: 0.00001
        During pre-training: Epochs: 50, batch size: 128
        Downstream tasks: Batch size: 32
        """

        # Define ECG model variables TODO: Add this to the argparser
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 18, ResidualBlock)
    
    elif args.ecg_model == "resnet34":
        ecg_model_name = "ResNet-34"
        print("ResNet-34 ECG model selected.")
        
        # Define ECG model variables TODO: Add this to the argparser
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 34, ResidualBlock)
    
    # Tokenize the text
    encoded_output = text_model.encode(dataset.X_train_text[:1000], add_special_tokens=True)

    # Embed the text
    embedded_output = text_model.embed(encoded_output)

    # Try sending through the ECG model
    with torch.no_grad():
        output = ecg_model(dataset.X_train_ecg_tensor)

    print(output.shape)
    print(embedded_output.shape)

    print("Done!")