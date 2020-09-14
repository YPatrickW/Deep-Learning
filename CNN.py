import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

'''
in_channel,out_channel = 5,10
width, height = 100,100
kernal_size =3
batch_size = 1
input = torch.randn(batch_size,in_channel,width,height)
conv_layer = torch.nn.Conv2d(in_channel,out_channel,kernel_size=kernal_size)
ouput = conv_layer(input)
input1 = [3,4,6,5,7,
          2,4,6,8,2,
          1,6,7,8,4,
          9,7,4,6,2,
          3,7,5,4,1
         ]
input1 = torch.Tensor(input1).view(1,1,5,5)#batch_size, channel, width, height
conv_layer1 = torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)#input channel, output channel kernel_Size
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)#output channel, input channel, width, height
conv_layer1.weight.data = kernel.data
output1 = conv_layer1(input1)
max_pooling = torch.nn.MaxPool2d(kernel_size=2)#步长也为2
output1 = max_pooling(output1)
'''
batch_size = 64
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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.conv1(x)
        z = F.relu(z)
        z = self.pooling(z)
        z = self.conv2(z)
        z = F.relu(z)
        z = self.pooling(z)
        out = z.view(batch_size, -1)
        out = self.fc(out)
        return out


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # GPU运算

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs,targets = inputs.to(device),targets.to(device)# 数据迁移
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
            images,labels = images.to(device),labels.to(device)#数据迁移到GPU上
            output = model(images)
            _, pred = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print("accuracy on test set :%d" % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
