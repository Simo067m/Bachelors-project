import torch
from torch import nn
from models.linear_projectors import EcgLinearProjectionHead

class ResidualBlock(nn.Module):
    """
    A residual block for a ResNet model.
    The residual block is used for the ResNet18 and ResNet34 models.
    For deeper models, a bottleneck block is used instead for computational efficiency.
    """
    def __init__(self, in_channels : int, out_channels : int, stride : int, downsample : nn.Module = None, expansion : int = 1):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # Expansion is 1 for ResNet18 and ResNet34

        # Define the layers of the residual block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm1d(out_channels * self.expansion) # Dimensions are expanded by the expansion factor deeper layer architectures
        self.relu2 = nn.ReLU(inplace=True)
    
    # Define the forward function
    def forward(self, x):
        # Save the input for the identity mapping
        identity = x

        # Forward pass through the first convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Forward pass through the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # If a downsampling model is passed in, perform downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add the identity to the output and apply the ReLU activation function
        out += identity
        out = self.relu2(out)
        return out

class BottleNeck(nn.Module):
    """
     A Bottleneck block for a ResNet model.
     The bottleneck block is used for the ResNet50, ResNet101, and ResNet152 models.
     However, it may be used for ResNet18 and ResNet34 as well, as in the ETP paper.
    """
    def __init__(self, in_channels : int, out_channels : int, stride : int, downsample : nn.Module = None, expansion : int = 4):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # Expansion is 4 for ResNet50, ResNet101, and ResNet152

        # Define the layers of the bottleneck block
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Save the input for the identity mapping
        identity = x

        # Forward pass through the first convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Forward pass through the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Forward pass through the third convolutional layer
        out = self.conv3(out)
        out = self.bn3(out)

        # If a downsampling model is passed in, perform downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add the identity to the output and apply the ReLU activation function
        out += identity
        out = self.relu3(out)
        return out

class ResNet(nn.Module):
    """
    ResNet modified for signal processing. The main modifications are the use of 1D layers instead of 2D layers.
    Follows the implementation as described in "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385).
    The architecture is built as in Table 1.
    Uses either a Residual Block or Bottleneck as "block", depending on depth.
    """
    def __init__(self, data_in_channels : int, num_classes : int, num_layers : int, block : object):
        super().__init__()
        """
        The number of layers in the network is specified by the num_layers parameter. Must match those from the paper, currently either 18 or 34.
        """
        if num_layers == 18:
            self.layers = [2, 2, 2, 2]
            self.expansion = 1 # 1 for ResNet18
            self.name = "ResNet-18"
        elif num_layers == 34:
            self.layers = [3, 4, 6, 3]
            self.expansion = 1 # 1 for ResNet34
            self.name = "ResNet-34"
        elif num_layers == 50:
            self.layers = [3, 4, 6, 3]
            self.expansion = 4
            self.name = "ResNet-50"
        elif num_layers == 101:
            self.layers = [3, 4, 23, 3]
            self.expansion = 4
            self.name = "ResNet-101"
        elif num_layers == 152:
            self.layers = [3, 8, 36, 3]
            self.expansion = 4
            self.name = "ResNet-152"
        else:
            raise ValueError("The number of layers must be either 18 or 34.")

        # Define the "stem" convolutional layer before the residual layers
        self.in_channels = 64 # in_channels defined by the paper, which can be modified for each layer by multiplication
        self.conv1 = nn.Conv1d(in_channels = data_in_channels, out_channels = self.in_channels, kernel_size=7, stride=2 , padding=3, bias=False) # The first layer has a kernel size of 7
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Define the maxpool layer
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Define the residual layers
        self.conv2_x = self._make_layer(block, 64, self.layers[0])
        self.conv3_x = self._make_layer(block, 128, self.layers[1], stride = 2)
        self.conv4_x = self._make_layer(block, 256, self.layers[2], stride = 2)
        self.conv5_x = self._make_layer(block, 512, self.layers[3], stride = 2)

        # Define the average pool layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Define the fully connected layer
        #self.fc = nn.Linear(512 * self.expansion, num_classes)
        self.linear_proj = EcgLinearProjectionHead(512 * self.expansion)

    def _make_layer(self, block, out_channels : int, blocks : int, stride : int = 1):
            downsample = None # Initialize the downsampling layer as None
            if stride != 1 or self.in_channels != out_channels * self.expansion:
                # If stride is not 1, then the dimensions of the input must be changed to match the output
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels * self.expansion, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm1d(out_channels * self.expansion)
                )
            layers = []
            # Create the first block
            layers.append(block(
                self.in_channels, out_channels, stride, downsample, self.expansion
            ))
            self.in_channels = out_channels * self.expansion

            # Create the remaining blocks
            for i in range(1, blocks):
                layers.append(block(self.in_channels, out_channels, stride = 1, expansion = self.expansion))
            return nn.Sequential(*layers)
        
    def forward(self, x):
            # Forward pass through the first convolutional layer
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Forward pass through the residual layers
            x = self.conv2_x(x)
            x = self.conv3_x(x)
            x = self.conv4_x(x)
            x = self.conv5_x(x)

            # Forward pass through the final layers
            x = self.avgpool(x)
            #x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            
            x = self.linear_proj(x) # Apply the linear projection head

            return x