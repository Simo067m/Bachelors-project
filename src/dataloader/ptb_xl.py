import pandas as pd
import numpy as np
import wfdb
import ast
import time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import random_split

# Seed for reproducibility (Current use: random_split)
seed = 42
torch.manual_seed = 42
# Define classes for loading each dataset
#TODO: Remove unnecessary attributes and functions

class ptb_xl_processor():
    """
    This class loads the PTB-XL dataset.
    https://physionet.org/content/ptb-xl/1.0.3/

    The functionality of this class is adapted from PhysioNet as well.

    Attributes:
    - X_train_ecg (list): Training ECG data
    - X_train_text (list): Training text data
    - y_train (list): Training targets
    - X_test_ecg (list): Testing ECG data
    - X_test_text (list): Testing text data
    - y_test (list): Testing targets
    """
    def __init__(self, path_to_dataset : str, sampling_rate : int = 100, split_method : str = "pre_split", test_fold : int = 10, val_fold : int = 9, load_raw : bool = False, load_meta : bool = False, single_label : bool = True):
        """
        Initializes a class instance, and loads the dataset. No function calls are necessary.

        Parameters:
        - path_to_dataset (str): The path to the folder containing the dataset.
        - sampling_rate (int): The sampling rate.
        - load_meta (bool): Whether to load the metadata.
        """

        self.split_method = split_method
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.single_label = single_label
        self.path_to_dataset = path_to_dataset
        self.sampling_rate = sampling_rate
        self.load_raw = load_raw

        if load_meta:
            self.meta = pd.read_csv(self.path_to_dataset+'ptbxl_database.csv', index_col='ecg_id')
        
        self.load_data()
    
    def load_data(self):
        """
        Loads the data.

        Returns:
        - X_train (list): Training data
        - y_train (list): Training targets
        - X_test (list): Testing data
        - y_test (list): Testing targets
        """

        if self.load_raw:
            # load and convert annotation data
            Y = pd.read_csv(self.path_to_dataset+'ptbxl_database.csv', index_col='ecg_id')
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

            # Apply diagnostic superclass
            Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

            if self.single_label:
                Y = self.make_single_label(Y)
            
            X_ecg, X_text = self.load_raw_data(Y)

            Y.to_csv("../Datasets/ptb-xl/data.csv", index=False)
            with open("../Datasets/ptb-xl/text_reports.txt", "w") as f:
                for report in X_text:
                    f.write(report + "\n")
            torch.save(torch.tensor(X_ecg, dtype=torch.float32), "../Datasets/ptb-xl/ecg_data.pt")
        
        else:
            Y = pd.read_csv(self.path_to_dataset + "saved_splits/data.csv")
            X_ecg = torch.load(self.path_to_dataset + "saved_splits/ecg_data.pt")
            with open(self.path_to_dataset + "saved_splits/text_reports.txt", "r") as f:
                X_text = f.readlines()
                X_text = [line.strip() for line in X_text]
            X_text = np.array(X_text)


        # Split into train and test using pytorch instead of the provided test_fold
        # Splits are chosen based on the split used in "Adversarial Spatiotemporal Contrastive Learning for Electrocardiogram Signals" (https://ieeexplore.ieee.org/document/10177892) Table 1.
        if self.split_method == "random_split":
            train_indices, val_indices, test_indices = random_split(np.arange(len(Y)), [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(seed))
            self.train_data, self.val_data, self.test_data = Y.iloc[train_indices], Y.iloc[val_indices], Y.iloc[test_indices]
            self.train_text, self.val_text, self.test_text = X_text[train_indices], X_text[val_indices], X_text[test_indices]
            self.train_ecg, self.val_ecg, self.test_ecg = X_ecg[train_indices], X_ecg[val_indices], X_ecg[test_indices]
        elif self.split_method == "stratified":
            train_indices, val_indices, test_indices = self.split_equal_distribution(Y)
            self.train_data, self.val_data, self.test_data = Y.iloc[train_indices], Y.iloc[val_indices], Y.iloc[test_indices]
            self.train_text, self.val_text, self.test_text = X_text[train_indices], X_text[val_indices], X_text[test_indices]
            self.train_ecg, self.val_ecg, self.test_ecg = X_ecg[train_indices], X_ecg[val_indices], X_ecg[test_indices]
        elif self.split_method == "pre_split":
            self.train_data, self.val_data, self.test_data = Y[(Y.strat_fold != self.test_fold) & (Y.strat_fold != self.val_fold)], Y[Y.strat_fold == self.val_fold], Y[Y.strat_fold == self.test_fold]
            self.train_text, self.val_text, self.test_text = X_text[np.where((Y.strat_fold != self.test_fold) & (Y.strat_fold != self.val_fold))], X_text[np.where(Y.strat_fold == self.val_fold)], X_text[np.where(Y.strat_fold == self.test_fold)]
            self.train_ecg, self.val_ecg, self.test_ecg = X_ecg[np.where((Y.strat_fold != self.test_fold) & (Y.strat_fold != self.val_fold))], X_ecg[np.where(Y.strat_fold == self.val_fold)], X_ecg[np.where(Y.strat_fold == self.test_fold)]

        # Save data
        self.train_data.to_csv(self.path_to_dataset + "saved_splits/train_data.csv", index=False)
        self.val_data.to_csv(self.path_to_dataset + "saved_splits/val_data.csv", index=False)
        self.test_data.to_csv(self.path_to_dataset + "saved_splits/test_data.csv", index=False)
        with open(self.path_to_dataset + "saved_splits/train_text.txt", "w") as f:
            for report in self.train_text:
                f.write(report + "\n")
        with open(self.path_to_dataset + "saved_splits/val_text.txt", "w") as f:
            for report in self.val_text:
                f.write(report + "\n")
        with open(self.path_to_dataset + "saved_splits/test_text.txt", "w") as f:
            for report in self.test_text:
                f.write(report + "\n")
        torch.save(self.train_ecg, self.path_to_dataset + "saved_splits/train_ecg.pt")
        torch.save(self.val_ecg, self.path_to_dataset + "saved_splits/val_ecg.pt")
        torch.save(self.test_ecg, self.path_to_dataset + "saved_splits/test_ecg.pt")

    def load_raw_data(self, df : pd.DataFrame):
        """
        Loads the raw signal data using the wfdb package.

        Parameters:
        - df (pandas.DataFrame): The dataset loaded as a pandas DataFrame.
        """
        if self.sampling_rate == 100:
            ecg_data = [wfdb.rdsamp(self.path_to_dataset+f) for f in tqdm(df.filename_lr, desc="Loading ECG data")]
            text_data = np.array([row for row in df.report])
        else:
            ecg_data = [wfdb.rdsamp(self.path_to_dataset+f) for f in df.filename_hr]
            text_data = np.array([row for row in df.report])
        ecg_data = np.array([signal for signal, meta in ecg_data])
        return ecg_data, text_data
    
    def aggregate_diagnostic(self, y_dic):
        """
        Calculates Aggregated diagnostics.
        """
        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.path_to_dataset+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    def make_single_label(self, data):
        """
        Removes rows with multiple and 0 labels from the dataset.
        """
        for idx, row in data.iterrows():
            if len(row.diagnostic_superclass) > 1 or len(row.diagnostic_superclass) == 0:
                data = data.drop(idx)

        return data

    def split_equal_distribution(self, data):
        """
        Splits the dataset into train, validation and test sets with equal class distribution.
        """
        targets = data.diagnostic_superclass
        train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        train_indices, test_indices = next(train_test_split.split(data, targets))
        val_indices, test_indices = next(val_test_split.split(data.iloc[test_indices], targets.iloc[test_indices]))

        return train_indices, val_indices, test_indices

class ptb_xl_dataset(Dataset):
    """
    This class implements pytorch's Dataset class for the PTB-XL dataset.
    It is used to load the data into a DataLoader.
    The data needs to be preprocessed before being loaded into this class.
    """
    def __init__(self, type : str, path_to_data : str, include_text : bool = False):

        self.include_text = include_text

        self.data = pd.read_csv(path_to_data+type+"_data.csv")
        self.ecg_data = torch.load(path_to_data+type+"_ecg.pt").clone().detach().permute(0, 2, 1)
        self.text_data = []
        with open(path_to_data+type+"_text.txt", "r") as f:
            for line in f:
                self.text_data.append(line.strip())
        
        # Convert text_data to a tensor for sending to gpu
        numerical_data = [[ord(char) for char in string] for string in self.text_data]
        self.tensors_on_gpu = [torch.tensor(data).cuda() for data in numerical_data]


        self.y = self.data.diagnostic_superclass
        self.y_tensor, self.y_tensor_one_hot = self.y_to_tensors(self.y)

    def y_to_tensors(self, y):
        """
        Converts the target data to tensors.
        """
        array = np.array(y)
        for i in range(len(array)):
            array[i] = ast.literal_eval(array[i])[0]
        
        self.label_map = {label: i for i, label in enumerate(sorted(set(array)))}
        indices = [self.label_map[label] for label in array]
        y_tensor = torch.tensor(indices)
        y_tensor_one_hot = torch.nn.functional.one_hot(torch.tensor(indices))

        return y_tensor, y_tensor_one_hot
    
    def __len__(self):
        # Required function for PyTorch Dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Required function for PyTorch Dataset
        # This function only returns the ECG data and the target, since the text encoder does not require training
        #return (self.X_ecg_tensor[idx], self.X_text[idx]), self.y_train_tensor[idx]
        if self.include_text:
            return self.ecg_data[idx], self.text_data[idx], self.y_tensor[idx]
        else:
            return self.ecg_data[idx], self.y_tensor[idx]
    
def ptb_xl_data_generator(configs, split_method : str = "pre_split", sampling_rate : int = 100, include_text : bool = False):
    """
    Generates the DataLoader objects for the PTB-XL dataset.
    """
    # Preprocess the data
    preprocessed = ptb_xl_processor(configs.path_to_dataset, split_method=split_method, sampling_rate = sampling_rate)
    # Load the data into the dataset
    train_dataset = ptb_xl_dataset("train", configs.path_to_splits, include_text=include_text)
    val_dataset = ptb_xl_dataset("val", configs.path_to_splits, include_text=include_text)
    test_dataset = ptb_xl_dataset("test", configs.path_to_splits, include_text=include_text)
    # Load the data into a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=True)
    return train_loader, val_loader, test_loader