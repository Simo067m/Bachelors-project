class Configs(object):
    """
    Configurations for the PTB-XL dataset.
    """
    def __init__(self):
        self.input_channels = 12 # Recordings are 12-lead ECGs
        self.num_classes = 5
        self.num_classes_disease = 2