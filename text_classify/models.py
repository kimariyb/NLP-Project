import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor):
        """
        x: [seq_len, batch_size, input_size]
        h0: [num_layers, batch_size, hidden_size]
        """
        x = x.unsqueeze(1) 
        # 1. 特征提取
        xn, hn = self.rnn(x, h0)
        xn = xn[-1]
        # 2. 全连接层
        out = self.fc(xn)
        # 3. softmax
        out = self.softmax(out)
        return out, hn
    
    def init_hidden(self):
        # 初始化隐藏层
        return torch.zeros(self.num_layers, 1, self.hidden_size)
    

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        """
        x: [seq_len, batch_size, input_size]
        h0: [num_layers, batch_size, hidden_size]
        c0: [num_layers, batch_size, hidden_size]
        """                    
        x = x.unsqueeze(1)
        xn, (hn, cn) = self.lstm(x, (h0, c0))
        xn = xn[-1]
        out = self.fc(xn)
        out = self.softmax(out)
        return out, (hn, cn)

    def init_hidden(self):
        # 初始化隐藏层
        return torch.zeros(self.num_layers, 1, self.hidden_size), torch.zeros(self.num_layers, 1, self.hidden_size)
    

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor, h0: torch.Tensor):
        """
        x: [seq_len, batch_size, input_size]
        h0: [num_layers, batch_size, hidden_size]
        """
        x = x.unsqueeze(1)
        xn, hn = self.gru(x, h0)
        xn = xn[-1]
        out = self.fc(xn)
        out = self.softmax(out)
        return out, hn

    def init_hidden(self):
        # 初始化隐藏层
        return torch.zeros(self.num_layers, 1, self.hidden_size)


if __name__ == '__main__':
    model = MyGRU(1, 1, 1, 1)
    print(model._get_name())