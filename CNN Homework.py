import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size = 64
test_input = torch.randn(1, 1, 28, 28)  # Batch_size, channel, width, height
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 均值，方差
])
train_data = datasets.MNIST("../dataset/mnist",
                            download=False, train=True, transform=transform)
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_Data = datasets.MNIST("../dataset/mnist",
                           download=False, train=False, transform=transform)
test_loader = DataLoader(dataset=test_Data, shuffle=False, batch_size=batch_size)


class Improved_CNN(torch.nn.Module):
    def __init__(self):
        super(Improved_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5, bias=False)  # 24x24
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5, bias=False)  # 20x20
        self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=5, bias=False, padding=2)  # 16x16
        self.Max_pooling = torch.nn.MaxPool2d(2)  # 默认步长为卷积核的大小，宽度和高度减小为原来的一半
        self.FC = torch.nn.Linear(120, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.conv1(x)  # 24x24
        z = F.relu(z)
        z = self.Max_pooling(z)  # 12x12
        z = self.conv2(z)  # 8x8
        z = F.relu(z)
        z = self.Max_pooling(z)  # 4x4
        z = self.conv3(z)  # 4x4
        z = F.relu(z)
        z = self.Max_pooling(z)  # 2x2
        out = z.view(batch_size, -1)
        return out


model = Improved_CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        # inputs,targets = inputs.to(device),targets.to(device)# 数据迁移
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss
        if batch_idx % 300 == 299:
            print("[%d,%5d] loss :%.3f" % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images,labels = images.to(device),labels.to(device)#数据迁移到GPU上
            output = model(images)
            _, pred = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print("accuracy on test set :%d" % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
