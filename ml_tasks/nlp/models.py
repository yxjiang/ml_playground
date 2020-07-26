import torch
import torch.nn as nn
import torch.nn.functional as F





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