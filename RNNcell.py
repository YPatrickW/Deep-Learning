import torch

'''
# RNN的数据集 (seq_len,batch_size,input_size)
# RNN cell : (input_size,hidden_size)
# input = (batch_size,input_size)
# hidden = (batch_size,hidden_size)
input_size = 4
batch_size = 1
seq_len = 3
hidden_size = 2
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)
for idx, input in enumerate(dataset):
    print("input_size= ", input.shape)
    hidden = cell(input, hidden)
    print("output_size: ", hidden.shape)
    print(hidden)
'''

#RNNcell
batch_size = 1
input_size = 4
hidden_size = 4
idx2char = ["e", "h", "l", "o"]
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x_one_hot = [one_hot[i] for i in x_data]
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)  # 长度x1


class model(torch.nn.Module):
    def __init__(self, batch_size, input_size, hidden_size):
        super(model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.RNNCell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.RNNCell(input, hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


model = model(input_size=input_size, batch_size=batch_size, hidden_size=hidden_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()
    print("pred_string= ",end="")
    for input, label in zip(inputs, labels):
        hidden = model(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()],end="")
    loss.backward()
    optimizer.step()
    print("   epoch: ", epoch + 1," loss: ", loss.item())


