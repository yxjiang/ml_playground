import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocabulary_size, config.word_embedding_length)
        self.hidden_size = 128
        self.num_layers = 1
        self.directions = 1
        self.rnn = nn.RNN(
            input_size=config.word_embedding_length, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers
        )
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=64)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(in_features=64, out_features=config.num_classes)

    def forward(self, x):
        batch = x.shape[0]
        x = self.embed(x)  # (batch, sentence_length, embedding_dim)
        x = x.permute(1, 0, 2).contiguous()  # (sentence_length, batch, embedding_dim)

        h0 = torch.zeros((self.num_layers * self.directions, batch, self.hidden_size)).to(self.config.device)
        output, ht = self.rnn(x, h0)
        ht = ht.permute(1, 0, 2)  # (batch, num_layer * directions, embedding_dim)
        ht = ht.contiguous().view(batch, self.num_layers * self.directions, self.hidden_size)
        x = F.relu(self.fc1(ht))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(batch, -1)
        return x



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