import torch
import torch.nn.functional as F


class Inception(torch.nn.Module):  # 构造Inception模块
    def __init__(self, in_channel):
        super(Inception, self).__init__()

        self.branch_pool = torch.nn.Conv2d(in_channel, 24, kernel_size=1)
        self.branch_1x1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)

        self.branch_5x5_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1, padding=1)
        self.branch_5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch_3x3_1 = torch.nn.Conv2d(in_channel, kernel_size=1)
        self.branch_3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)  # 平均池化
        branch_pool = self.branch_pool(branch_pool)  # 池化后卷积
        branch_1x1 = self.branch_1x1(x)

        branch_5x5_1 = self.branch_5x5_1(x)
        branch_5x5_2 = self.branch_5x5_2(branch_5x5_1)

        branch_3x3_1 = self.branch_3x3_1(x)
        branch_3x3_2 = self.branch_3x3_2(branch_3x3_1)
        branch_3x3_3 = self.branch_3x3_3(branch_3x3_2)

        output = [branch_1x1, branch_5x5_2, branch_3x3_3, branch_pool]
        return torch.cat(output, dim=1)


class Google_Net(torch.nn.Module):
    def __init__(self):
        super(Google_Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88,20,kernel_size=5)

        self.incep1 = Inception(in_channel=10)
        self.incep2 = Inception(in_channel=20)

        self.Max_pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(1408,10)

    def forward(self,x):
        in_size = x.shape[0]
        z = self.conv1(x)
        z = self.Max_pooling(z)
        z = F.relu(z)
        z = self.incep1(z)
        z = self.conv2(z)
        z = self.Max_pooling(z)
        z = self.incep2(z)
        z = z.view(in_size,-1)
        return z