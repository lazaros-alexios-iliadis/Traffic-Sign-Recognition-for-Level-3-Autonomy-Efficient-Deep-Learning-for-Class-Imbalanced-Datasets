import torch.nn as nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1_in_features = 256 * 7 * 7
        self.fc1 = nn.Linear(self.fc1_in_features, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)

        return x


class MicroVGG(nn.Module):
    """
    VGG like model, inspired from https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, num_classes: int, input_channels: int = 3, hidden_units: int = 64):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial size by half
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Further reduces spatial size by half
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Further reduces spatial size by half
        )

        # Adaptive Pooling to ensure a fixed output size
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output size: (batch, hidden_units, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flattens (batch, hidden_units, 1, 1) to (batch, hidden_units)
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.global_pool(x)  # Ensures consistent output size
        x = self.classifier(x)
        return x


# Define Optimized MicroVGG Model with More Blocks
class ProposedCustomCNN(nn.Module):
    def __init__(self, num_classes=43, hidden_units=64):
        super().__init__()

        # Depthwise Separable Convolution Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, hidden_units, kernel_size=3, stride=2, groups=1, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, groups=hidden_units, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU()
        )

        # Depthwise Separable Convolution Block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=2, groups=hidden_units, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU()
        )

        # Depthwise Separable Convolution Block 3
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, groups=hidden_units, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU()
        )

        # Additional Convolution Block 4
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, groups=hidden_units, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU()
        )

        # Additional Convolution Block 5
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, groups=hidden_units, padding=1),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=1),
            nn.BatchNorm2d(hidden_units),
            nn.GELU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_units, num_classes)
        # self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x
