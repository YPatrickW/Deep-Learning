import torch
import torch.nn.functional as F


class Residual_Block(torch.nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.channels = channels  # 保持通道数不变才能相加
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        z = self.conv1(x)
        z = F.relu(z)
        y = self.conv2(z)
        z = z + y  # 先相加，后激活
        z = F.relu(z)
        return z


class Residual_Net(torch.nn.Module):
    def __init__(self):
        super(Residual_Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.Max_pooling = torch.nn.MaxPool2d(kernel_size=2)

        self.residual_block1 = Residual_Block(channels=16)
        self.residual_block2 = Residual_Block(channels=32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.shape[0]
        z = self.conv1(x)
        z = F.relu(z)
        z = self.Max_pooling(z)
        z = self.residual_block1(z)
        z = self.Max_pooling(x)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.Max_pooling(z)
        z = self.residual_block2
        z = z.view(in_size, -1)
        z = self.fc(z)
        return z
