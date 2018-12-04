import torch
import torch.nn as nn
class MYGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers,dropout):
        super(MYGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.last_fully_connected = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_batch):
        output, (hidden, cell_state) = self.gru(input_batch)
        output = output[:,-1,:]
        output = self.last_fully_connected(output)
        output = self.softmax(output)
        return output

