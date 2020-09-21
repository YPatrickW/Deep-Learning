import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn.utils

USE_GPU = False
batch_size = 256
hidden_size = 100
n_layer = 2
n_epoch = 100
n_chars = 128  # 转换为ASCII码


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = "./NameData/names_train.csv/names_train.csv" if is_train_set else "./NameData/names_test.csv/names_test.csv"
        train_data = pd.read_csv(filename, header=None)
        self.names = train_data[0]
        self.len = len(train_data[1])
        self.countries = train_data[1]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.country_Dict()
        self.country_num = len(self.country_list)

    def __getitem__(self, item):
        return self.names[item], self.country_dict[self.countries[item]]

    def __len__(self):
        return self.len

    def country_Dict(self):
        country_dict = {}
        for idx, country in enumerate(self.countries, 0):
            country_dict[country] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def ger_country_name(self):
        return self.country_num


Train_Dataset = NameDataset(is_train_set=True)
Test_Dataset = NameDataset(is_train_set=False)
Train_loader = DataLoader(dataset=Train_Dataset, batch_size=batch_size, shuffle=True)
Test_loader = DataLoader(dataset=Test_Dataset, batch_size=batch_size, shuffle=False)

Country_num = Test_Dataset.ger_country_name()


class RNNClassifer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, bidirectional=True, ):
        super(RNNClassifer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_directions = 2 if bidirectional else 1
        self.layers = num_layers
        self.batch_size = batch_size
        self.embedding = torch.nn.Embedding(input_size,
                                            hidden_size)  # 输入为（seq_len,batch_size）,输出为（seq_len,batch_size,hidden_size）
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def init_hidden(self):
        hidden = torch.zeros(self.layers * self.n_directions, self.batch_size, self.hidden_size)
        return torch.Tensor(hidden)

    def forward(self, input, seq_length):
        input = input.t()
        batch_size = input.shape[0]
        hidden = self.init_hidden()
        embedding = self.embedding(input)  # （seq_len,batch_size,hidden_size）
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_length)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output


def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)


def creat_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def name2tensor(names, countries):
    name_sequences_and_lengths = [name2list(name) for name in names]
    name_seq = [name_c[0] for name_c in name_sequences_and_lengths]
    seq_length = [name_c[1] for name_c in name_sequences_and_lengths]
    seq_length = torch.LongTensor(seq_length)
    countries = countries.long()
    seq_tensor = torch.zeros(len(name_seq), seq_length.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_seq, seq_length), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    seq_length, perm_idx = seq_length.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    return creat_tensor(seq_tensor), \
           creat_tensor(seq_length), \
           creat_tensor(countries)


classifier = RNNClassifer(input_size=n_chars, hidden_size=hidden_size, num_layers=n_layer, bidirectional=True,
                          output_size=18, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)


def train_model():
    total_loss = 0
    for idx, (names, countries) in enumerate(Train_loader, 1):
        inputs, seq_lengths, target = name2tensor(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if idx % 10 == 0:
            print(f"loss = {total_loss / (idx * len(inputs))}")
    return total_loss


if __name__ == '__main__':
    train_model()
