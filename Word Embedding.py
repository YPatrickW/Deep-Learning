import torch

batch_size = 1
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layer = 2
seq_len = 5
idx2char = ["e", "h", "l", "o"]
x_data = [[1, 0, 2, 2, 3]]  # hello 维度为（batch，seq_len ）
y_data = [3, 1, 2, 3, 2]  # ohlol 维度为batch_size x seq_len
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


class model(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, input_size, num_layer, embedding_size, num_class):
        super(model, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layer = num_layer
        self.embedding_size = embedding_size
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.num_class = num_class  # 分类问题的种类
        self.RNN = torch.nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layer)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_class)

    def forward(self, input):
        hidden = torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
        x = self.emb(input)  # 词嵌入
        out, hn = self.RNN(x, hidden)  # 返回h1--hn和单独的一个hn
        out = self.fc(out)
        return out.view(-1, self.num_class)  # (seq_len x batch_size,num_class)
