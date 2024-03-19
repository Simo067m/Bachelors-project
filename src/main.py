# Import packages
import argparse
import os

# Import custom modules
from dataloader import ptb_xl_dataset
from models.clinical_bert import bio_clinical_BERT
from models.resnet18 import resnet18

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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load the dataset

    if args.dataset == "ptb-xl":
        print("PTB-XL dataset selected.")
        
        # Define dataset variables TODO: Add this to the argparser
        path_to_dataset = os.path.join(os.getcwd(), "Datasets", "ptb-xl", "")
        sampling_rate = 100
        test_fold = 10
        dataset = ptb_xl_dataset(path_to_dataset = path_to_dataset, sampling_rate = sampling_rate, test_fold = test_fold)
    
    # Load the text model
        
    if args.text_model == "bio-clinical-bert":
        print("BioClinicalBERT model selected.")
        
        # Define text model variables TODO: Add this to the argparser
        bio_clinical_bert = bio_clinical_BERT()

    # Load the ECG model
        
    if args.ecg_model == "resnet18":
        print("ResNet18 model selected.")
        
        # Define ECG model variables TODO: Add this to the argparser
        resnet = resnet18()
    
    # Tokenize the text
    encoded_output = bio_clinical_bert.encode(dataset.X_train_text[:50], add_special_tokens=True)

    # Embed the text
    embedded_output = bio_clinical_bert.embed(encoded_output)

    print(embedded_output)