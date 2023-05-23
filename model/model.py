import torch
import torch.nn as nn

class FaceRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognizer, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 2,
            kernel_size = 3,
            dtype = torch.float64
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels = 2,
            out_channels = 2,
            kernel_size = 3,
            dtype = torch.float64
        )
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.conv3 = nn.Conv2d(
            in_channels = 2,
            out_channels = 3,
            kernel_size = 3,
            dtype = torch.float64
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels = 3,
            out_channels = 3,
            kernel_size = 3,
            dtype = torch.float64
        )
        self.relu4 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.dense1 = nn.Linear(
            in_features = 6627,
            out_features = 3000,
            dtype = torch.float64
        )

        self.relu5 = nn.ReLU()

        self.dense2 = nn.Linear(
            in_features = 3000,
            out_features = 1000,
            dtype = torch.float64
        )

        self.relu6 = nn.ReLU()

        self.dense3 = nn.Linear(
            in_features = 1000,
            out_features = num_classes,
            dtype = torch.float64
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.maxpool1(y)
        y = self.conv3(y)
        y = self.relu3(y)
        y = self.conv4(y)
        y = self.relu4(y)
        y = self.maxpool2(y)
        y = y.reshape(-1, y.shape[1] * y.shape[2] * y.shape[3])
        y = self.dense1(y)
        y = self.relu5(y)
        y = self.dense2(y)
        y = self.relu6(y)
        y = self.dense3(y)
        return y