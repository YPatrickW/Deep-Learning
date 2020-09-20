import torch

'''  
# RNN : (input_size,hidden_size,num_layers)
cell = torch.nn.RNN(input_size,hidden_size,num_layers)
input = (seq_len,batch_size,input_size)
input 是整个序列
hidden = (num_layers,batch_size,hidden_size)
out,hidden = cell(input,hidden)#自动循环
out = (seq_len,batch_size,hidden_size)
out:最上层的h1--hn，hidden：最后的hn
'''

'''
batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layer = 1
cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layer)
inputs = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(num_layer,batch_size,hidden_size)
out,hidden = cell(inputs,hidden)
print("output size :",out.shape)
print("output: ",out)
print("hidden size : ",hidden.shape)
print("hidden: ",hidden)
'''

batch_size = 1
input_size = 4
hidden_size = 4
num_layer = 1
seq_len = 5
idx2char = ["e", "h", "l", "o"]
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x_one_hot = [one_hot[i] for i in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)  # seq_len x batch_size x 1
labels.reshape((seq_len * batch_size), 1)  # seq_len x batch_size x 1


class model(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, input_size, num_layer):
        super(model, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layer = num_layer
        self.RNN = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer)

    def forward(self, input):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        out, hn = self.RNN(input, hidden)  # 返回h1--hn和单独的一个hn
        return out.view(-1, self.hidden_size)  # (seq_len x batch_size,1)


model = model(input_size=input_size, hidden_size=hidden_size, num_layer=num_layer, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(15):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    _, idx = output.max(dim=1)
    idx = idx.data.numpy()
    print("predict string: ", "".join([idx2char[x] for x in idx]), end="")
    print("   epoch: ", epoch + 1, " loss: ", loss.item())
