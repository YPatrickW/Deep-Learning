import torch
'''
# RNN的数据集 （seq_len,batch_size,input_size）
# RNN cell : (input_size,hidden_size)
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
'''  
# RNN : (input_size,hidden_size,num_layers)
cell = (input_size,hidden_size,num_layers)
input = (seq_len,batch_size,input_size)
hidden = (num_layers,batch_size,hidden_size)
out,hidden = cell(input,hidden)#自动循环
out:最上层的Rn，hidden：最后的hn
'''
