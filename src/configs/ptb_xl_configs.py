import os

class Configs(object):
    """
    Configurations for the PTB-XL dataset.
    """
    def __init__(self):
        self.in_channels = 12 # Recordings are 12-lead ECGs
        self.num_classes = 5
        self.num_classes_disease = 2
    
        self.path_to_dataset = os.path.join(os.getcwd(), "Datasets", "ptb-xl", "")
        self.sampling_rate = 100
        self.test_fold = 10