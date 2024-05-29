# Import packages
import argparse
import os
import torch
import datetime

# Import custom modules
from dataloader.ptb_xl import ptb_xl_data_generator
from models.clinical_bert import bio_clinical_BERT
from models.resnet import ResNet, ResidualBlock
from models.resnet2D import ModifiedResNet1D
from models.linear_classifier import LinearClassifier
from train_eval.trainer import Trainer
from train_eval.loss import NT_Xent_loss

# Remove irrelevant pytorch storage warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)

def parse_args():

    parser = argparse.ArgumentParser(description='Representation learning of multimodal data using multiview machine learning')

    parser.add_argument("-load_raw_data", action="store_true", default=False, help="Load raw data")

    data_split_method = parser.add_mutually_exclusive_group()
    data_split_method.add_argument("-pre-split", action="store_const", help="Use pre-split data", dest="data_split_method", const="pre_split")
    
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument("-ptb-xl", action="store_const", help="PTB-XL dataset", dest="dataset", const="ptb-xl")

    text_model = parser.add_mutually_exclusive_group()
    text_model.add_argument("-bioclinicalbert", action="store_const", help="BioClinicalBERT model", dest="text_model", const="bioclinicalbert")

    ecg_model = parser.add_mutually_exclusive_group()
    ecg_model.add_argument("-resnet18", action="store_const", help="ResNet18 model", dest="ecg_model", const="resnet18")
    ecg_model.add_argument("-resnet34", action="store_const", help="ResNet34 model", dest="ecg_model", const="resnet34")
    ecg_model.add_argument("-resnet18-bottleneck", action="store_const", help="ResNet18 with bottleneck model", dest="ecg_model", const="resnet18-bottleneck")

    parser.add_argument("-log-wandb", action="store_true", default=False, help="Log data to wandb", dest="log_wandb")
    parser.add_argument("-wandb-project", type=str, help="Wandb project name")

    parser.add_argument("-run-config", nargs="+", action=StoreDictKeyPair, metavar="KEY=VALUE", help="Run configurations", dest="run_config") # Use this like: -run_config param1=value1 param2=value2
    
    """
    For the run configs, the following should be included:
    - task: ECG_pre_training, linear_classifier
    - epochs: int
    - save_name: str to save the model
    - batch_size: int
    - pre_trained_ecg_model: str
    """

    return parser.parse_args()

def validate_args(args):
    if not any([args.dataset, args.text_model, args.ecg_model]):
        raise ValueError("At least one option should be selected.")

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Initialize an empty dictionary
        config_dict = {}
        
        # Process each key=value pair
        for item in values:
            key, value = item.split("=")
            config_dict[key] = value
        
        # Set the dictionary as an attribute on the namespace object
        setattr(namespace, self.dest, config_dict)

if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    # Set the device to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {device}")
    print(f"Load raw data? {args.load_raw_data}")

    # Load the dataset

    if args.dataset == "ptb-xl":
        dataset_name = "PTB-XL"
        
        # Import configs
        from configs.ptb_xl_configs import Configs
        configs = Configs(int(args.run_config["batch-size"]))

        # Define dataset variables
        if args.run_config["task"] == "ECG_pre_training":
            train_loader, val_loader, test_loader = ptb_xl_data_generator(configs, split_method=args.data_split_method, use_translated=True, load_raw_data=args.load_raw_data, include_text=True)
        else:
            train_loader, val_loader, test_loader = ptb_xl_data_generator(configs, split_method=args.data_split_method, use_translated=True, load_raw_data=args.load_raw_data)
        
    
    # Load the text model
        
    if args.text_model == "bioclinicalbert":
        text_model_name = "BioClinicalBERT"
        
        # Define text model variables
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

        # Define ECG model variables
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 18, ResidualBlock).to(device)
    
    elif args.ecg_model == "resnet34":
        ecg_model_name = "ResNet-34"
        
        # Define ECG model variables
        ecg_model = ResNet(configs.in_channels, configs.num_classes, 34, ResidualBlock).to(device)
    
    elif args.ecg_model == "resnet18-bottleneck":
        ecg_model_name = "ResNet-18-BottleNeck"

        # Define ECG model variables
        ecg_model = ModifiedResNet1D(layers=[2, 2, 2, 2], output_dim=configs.num_classes, heads=8, input_resolution=1000, width=64).to(device)

    # Specify the wandb configurations
    if args.log_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_config['task']}_ECG_{ecg_model_name}_Text_{text_model_name}_{dataset_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            config={
                "learning_rate" : configs.learning_rate,
                "weight_decay" : configs.weight_decay,
                "ECG_model" : ecg_model_name,
                "text_model" : text_model_name,
                "dataset" : dataset_name,
                "epochs" : args.run_config["epochs"],
                "architecture" : "ETP"
            }
        )
        print(f"Logging to wandb project {args.wandb_project}.")
    
    # Print the run configurations
    print(f"Selected task: {args.run_config['task']}")
    print(f"Selected epochs: {args.run_config['epochs']}")
    # Make save_name "None" if not specified
    if "save_name" not in args.run_config:
        args.run_config["save_name"] = None # Does not save a model if the save name is of type None
    print(f"Save name: {args.run_config['save_name']}")
    
    # Print names of the selected models
    print(f"Selected dataset: {dataset_name}")
    print(f"Selected text model: {text_model_name}")
    print(f"Selected ECG model: {ecg_model_name}")

    # Run the task

    trainer = Trainer()

    if args.run_config["task"] == "ECG_pre_training":
        print(f"Running ECG pre-training for {args.run_config['epochs']} epochs.")

        # Define the optimizer and criterion
        optimizer = torch.optim.Adam(ecg_model.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
        criterion = NT_Xent_loss(configs.batch_size)

        # Train the model
        losses, val_losses = trainer.ecg_encoder_pre_train(ecg_model, text_model, train_loader, val_loader, int(args.run_config["epochs"]), optimizer, criterion, device, save_name = args.run_config["save_name"])

        # Evaluate the model
        avg_similarity, accuracy = trainer.evaluate_ecg_encoder(ecg_model, text_model, test_loader, device)
    
    elif args.run_config["task"] == "train_linear_classifier":
        print(f"Training linear classifier for {args.run_config['epochs']} epochs.")
        print(f"Using pre-trained ECG model: {args.run_config['pre_trained_ecg_model']}")

        # Load the pre-trained ECG model
        ecg_model.load_state_dict(torch.load("saved_models/"+args.run_config["pre_trained_ecg_model"]))

        # Define the classifier
        classifier = LinearClassifier(configs).to(device)

        # Define the optimizer and criterion
        optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.learning_rate, weight_decay=configs.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # Train the model
        losses, val_losses = trainer.train_linear_classifier(ecg_model, classifier, train_loader, val_loader, int(args.run_config["epochs"]), optimizer, criterion, device, save_name = args.run_config["save_name"])

        # Evaluate the model
        accuracy, f1_score = trainer.test_linear_classifier(ecg_model, classifier, test_loader, device)

    print("Done!")