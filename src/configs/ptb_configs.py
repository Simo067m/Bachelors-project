import os

class Configs(object):
    """
    Configurations for the PTB-XL dataset.
    """
    def __init__(self, batch_size):
        self.in_channels = 12 # Recordings are 12-lead ECGs
        self.num_classes = 5
        self.num_classes_disease = 2
    
        self.path_to_dataset = os.path.join(os.getcwd(), "Datasets", "ptb", "")
        #self.path_to_dataset = os.path.join("c:\\", "Users", "ssjsi", "Documents", "Bachelors-project", "Datasets", "ptb", "")

        self.learning_rate = 0.002 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2
        self.weight_decay = 0.00001 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2
        self.batch_size = batch_size # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2