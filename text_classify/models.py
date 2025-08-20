import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(-1)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [batch_size, seq_len, input_size]
        # lengths: [batch_size] 每个序列的实际长度
        # pack_padded_sequence 将输入序列进行打包，以减少不必要的计算
        packed_x = pack_padded_sequence(
            input=x, 
            lengths=lengths.cpu(),
            batch_first=True
        ) # [batch_size, seq_len, input_size] 
        out, _ = self.rnn(packed_x) # [batch_size, seq_len, hidden_size]

        # pad_packed_sequence 将打包后的序列进行解包，以恢复原始的序列长度
        out, _ = pad_packed_sequence(out, batch_first=True) # [batch_size, seq_len, hidden_size]
        last_out = out[torch.arange(out.size(0)), lengths - 1] # [batch_size, hidden_size]
        out = self.fc(last_out)
        out = self.softmax(out)
        return out


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: [batch_size, seq_len, input_size]
        lengths: [batch_size] 每个序列的实际长度
        """                    
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        out, _ = self.lstm(packed_x)

        out, _ = pad_packed_sequence(out, batch_first=True)
        last_out = out[torch.arange(out.size(0)), lengths - 1]
        out = self.fc(last_out)
        out = self.softmax(out)
        return out
    

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: [batch_size, seq_len, input_size]
        lengths: [batch_size] 每个序列的实际长度
        """
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        out, _ = self.gru(packed_x)

        out, _ = pad_packed_sequence(out, batch_first=True)
        last_out = out[torch.arange(out.size(0)), lengths - 1]
        out = self.fc(last_out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    model = MyGRU(1, 1, 1, 1)
    print(model._get_name())