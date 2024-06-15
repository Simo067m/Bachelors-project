import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from configs.ptb_configs import Configs
import wfdb
import ast
from torch.utils.data import random_split

class PTB(Dataset):
    """
    No need for splitting, since this entire dataset is for testing.
    """
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset

        self.data, self.targets = self.load_data()

        self.targets_tensor = self.y_to_tensors(self.targets)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets_tensor[idx]

    def load_data(self):
        file_path = self.path_to_dataset + 'RECORDS'

        with open(file_path, 'r') as file:
            lines = file.readlines()
        records = [line.strip() for line in lines]

        data = [wfdb.rdsamp(self.path_to_dataset + record) for record in records]

        remove = ["Valvular heart disease", "Heart failure", "Palpitation", "Stable angina", "Hypertrophy", "Unstable angina", "Myocarditis"]
        #remove = []

        targets = []
        ecg = []
        for i in range(len(data)):
            target = data[i][1]["comments"][4].split(":")[1].strip()
            if target != "n/a":
                if target not in remove:
                    if target[:5] == "Heart":
                        if target.split(" ")[0] == "Heart":
                            continue
                            targets.append(target)
                            ecg.append(torch.tensor(data[i][0][:32000, :12], dtype=torch.float32))
                    else:
                        targets.append(target)
                        ecg.append(torch.tensor(data[i][0][:32000, :12], dtype=torch.float32))
        ecg_data = torch.stack(ecg)

        # Define the segment length
        segment_length = 1000
        amount = len(ecg_data)
        samples = 32000
        # Lists to hold the segmented data and their corresponding targets
        segmented_data = []
        segmented_targets = []

        for idx in range(amount):
            # Extract the current ECG record and its target
            ecg_record = ecg_data[idx]
            target = targets[idx]
            
            # Split the current ECG record into segments
            for i in range(0, samples - segment_length + 1, segment_length):
                segment = ecg_record[i:i + segment_length]
        
                # Append segment and its corresponding target to lists
                segmented_data.append(segment)
                segmented_targets.append(target)

        # Convert lists to tensors
        segmented_data = torch.stack(segmented_data).permute(0, 2, 1)
        #segmented_targets = torch.stack(segmented_targets)

        counts = {}

        # Iterate over the list and count occurrences
        for item in segmented_targets:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        
        print(counts)

        return segmented_data, segmented_targets
    
    def y_to_tensors(self, y):
        """
        Converts the target data to tensors.
        """
        
        self.label_map = {label: i for i, label in enumerate(sorted(set(y)))}
        indices = [self.label_map[label] for label in y]
        y_tensor = torch.tensor(indices)
        y_tensor_one_hot = torch.nn.functional.one_hot(torch.tensor(indices))
        print(set(y_tensor).__len__())
        return y_tensor

def ptb_data_generator(configs):
    dataset = PTB(configs.path_to_dataset)
    train, val, test = random_split(dataset, [0.8, 0.1, 0.1], torch.Generator().manual_seed(43))
    #dataloader = DataLoader(dataset, batch_size = configs.batch_size, shuffle = True)

    train = DataLoader(train, batch_size = configs.batch_size, shuffle = True, drop_last=True)
    val = DataLoader(val, batch_size = configs.batch_size, shuffle = True, drop_last=True)
    test = DataLoader(test, batch_size = configs.batch_size, shuffle = True, drop_last=True)
    return train, val, test