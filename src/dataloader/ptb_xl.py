import pandas as pd
import numpy as np
import wfdb
import ast
import time
from tqdm import tqdm
import torch

# Define classes for loading each dataset

class ptb_xl_dataset:
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
    def __init__(self, path_to_dataset : str, sampling_rate : int, test_fold : int, load_meta = False):
        """
        Initializes a class instance, and loads the dataset. No function calls are necessary.

        Parameters:
        - path_to_dataset (str): The path to the folder containing the dataset.
        - sampling_rate (int): The sampling rate.
        - test_fold (int): The number of.
        - load_meta (bool): Whether to load the metadata.
        """
        # Check loading time
        start_time = time.time()

        self.path_to_dataset = path_to_dataset
        self.sampling_rate = sampling_rate
        self.test_fold = test_fold
        if load_meta:
            self.meta = pd.read_csv(self.path_to_dataset+'ptbxl_database.csv', index_col='ecg_id')
        self.X_train_ecg, self.X_train_text, self.y_train, self.X_test_ecg, self.X_test_text, self.y_test = self.load_data()
        # Save ecg data as tensors
        self.X_train_ecg_tensor = torch.tensor(self.X_train_ecg, dtype=torch.float32)
        self.X_test_ecg_tensor = torch.tensor(self.X_test_ecg, dtype=torch.float32)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Finished loading PTB-XL in {minutes} minutes and {seconds} seconds.")
    
    def load_data(self):
        """
        Loads the data.

        Returns:
        - X_train (list): Training data
        - y_train (list): Training targets
        - X_test (list): Testing data
        - y_test (list): Testing targets
        """
        # load and convert annotation data
        Y = pd.read_csv(self.path_to_dataset+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X_ecg, X_text = self.load_raw_data(Y)

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

        # Split data into train and test
        test_fold = self.test_fold

        # Train
        X_train_ecg = X_ecg[np.where(Y.strat_fold != test_fold)]
        X_train_text = X_text[np.where(Y.strat_fold != test_fold)].tolist()
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass

        # Test
        X_test_ecg = X_ecg[np.where(Y.strat_fold == test_fold)]
        X_test_text = X_text[np.where(Y.strat_fold == test_fold)].tolist()
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        return X_train_ecg, X_train_text, y_train, X_test_ecg, X_test_text, y_test

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
            text_data = [row for row in df.report]
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