import os

class Configs(object):
    """
    Configurations for the PTB-XL dataset.
    """
    def __init__(self, use_translated : bool = True):
        self.in_channels = 12 # Recordings are 12-lead ECGs
        self.num_classes = 5
        self.num_classes_disease = 2
    
        self.path_to_dataset = os.path.join(os.getcwd(), "Datasets", "ptb-xl", "")
        self.sampling_rate = 100

        self.learning_rate = 0.002 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2
        self.weight_decay = 0.00001 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2
        self.batch_size = 128 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2

class TestingConfigs(object):
    """
    Configurations for the PTB-XL dataset.
    """
    def __init__(self):
        self.in_channels = 12 # Recordings are 12-lead ECGs
        self.num_classes = 5
        self.num_classes_disease = 2
    
        self.path_to_dataset = os.path.join("c:\\", "Users", "ssjsi", "Documents", "Bachelors-project", "Datasets", "ptb-xl", "")
        self.sampling_rate = 100

        self.learning_rate = 0.002 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2
        self.weight_decay = 0.00001 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2
        self.batch_size = 128 # From "ETP: Learning Transferable ECG Representations via ECG-Text Pre-Training" (https://arxiv.org/pdf/2309.07145.pdf) section 3.2