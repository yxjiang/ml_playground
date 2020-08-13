import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocabulary_size, config.word_embedding_length)
        self.conv_layer_sizes = config.conv_layer_sizes

        for i, size in enumerate(self.conv_layer_sizes):
            self.add_module("conv" + str(i), nn.Conv2d(1, 1, kernel_size=(size, self.config.word_embedding_length)).to(self.config.device))
            self.add_module("pool" + str(i), nn.MaxPool2d((config.sentence_max_length - size + 1, 1)).to(self.config.device))

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(self.conv_layer_sizes), config.num_classes)


    def forward(self, x):
        batch = x.shape[0]
        x = torch.unsqueeze(self.embed(x), 1)  # [NCHW], add channel to dimension 1
        # convs
        xs = []
        for i in range(len(self.conv_layer_sizes)):
            xs.append(self.config.activation(self._modules["conv" + str(i)](x)))  # conv modules
            xs[i] = self._modules["pool" + str(i)](xs[i])  # max over time pooling modules

        x = torch.cat(xs, dim=-1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(batch, -1)
        
        return x


class RNN(nn.Module):
    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.config = config
        self.rnn_type = config.rnn_type
        self.embed = nn.Embedding(vocabulary_size, config.word_embedding_length)
        self.num_layers = config.num_layers if config.num_layers is not None else 1
        self.directions = config.num_directions if config.num_directions is not None else 1
        self.hidden_size = 128
        if config.rnn_type is nn.RNN:
            self.rnn = nn.RNN(
                input_size=config.word_embedding_length, 
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                bidirectional=False if self.directions == 1 else True
            )
        elif config.rnn_type is nn.LSTM:
            self.cell_size = 128
            self.rnn = nn.LSTM(
                input_size=config.word_embedding_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False if self.directions == 1 else True
            )
        elif config.rnn_type is nn.GRU:
            self.rnn = nn.GRU(
                input_size=config.word_embedding_length, 
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                bidirectional=False if self.directions == 1 else True
            )
        self.fc1 = nn.Linear(in_features=self.hidden_size * self.directions, out_features=64)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(in_features=64, out_features=config.num_classes)

    def forward(self, x, x_lens):
        batch = x.shape[0]
        x = self.embed(x)  # (batch, sentence_length, embedding_dim)
        x = x.to(self.config.device)
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        h0 = torch.zeros((self.num_layers * self.directions, batch, self.hidden_size)).to(self.config.device)
        if self.rnn_type is nn.RNN or self.rnn_type is nn.GRU:
            output, ht = self.rnn(x, h0)
        elif self.rnn_type is nn.LSTM:
            c0 = torch.zeros((self.num_layers * self.directions, batch, self.hidden_size)).to(self.config.device)
            output, (ht, ct) = self.rnn(x, (h0, c0))
        # decompose layers and directions, (layer, directions, batch, embedding_dim)
        ht = ht.view(self.num_layers, self.directions, ht.shape[1], -1) 
        ht = ht[-1]  # get last layer and remove layer dimension, (directions, batch, embedding_dim)
        ht = ht.permute(1, 0, 2)  # (batch, directions, embedding_dim)
        ht = ht.contiguous().view(batch, -1)
        x = F.relu(self.fc1(ht))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(batch, -1)
        return x


class PositionEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        multiplier = (torch.arange(0, dim_model, 2).float() / dim_model) * -torch.log(10000.0)
        pe[:, 0::2] = torch.sin(position * multiplier)
        pe[:, 1::2] = torch.cos(position * multiplier)
        pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, dim_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer:
    def __init__(self):
        super(Transformer, self).__init__()
