import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.out_channel = config.out_channel
        self.conv_w3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, config.word_embedding_dim))
        self.conv_w4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, config.word_embedding_dim))
        self.conv_w4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, config.word_embedding_dim))
        self.max_pool_w3 = nn.MaxPool2d(kernel_size=(self.config.sentence_max_size - 3 + 1, 1))
        self.max_pool_w4 = nn.MaxPool2d(kernel_size=(self.config.sentence_max_size - 4 + 1, 1))
        self.max_pool_w5 = nn.MaxPool2d(kernel_size=(self.config.sentence_max_size - 5 + 1, 1))
        self.linear = nn.Linear(3, config.label_size)


    def forward(self, x):
        batch = 1
        # conv and pool the inputs with different sizes
        x_w3 = self.max_pool_w3(F.relu(self.conv_w3(x)))
        x_w4 = self.max_pool_w4(F.relu(self.conv_w4(x)))
        x_w5 = self.max_pool_w5(F.relu(self.conv_w5(x)))

        # concatenate the extracted features
        x = torch.cat((x_w3, x_w4, x_w5), -1)
        x = x.view(batch, 1, -1)

        # project to labels
        x = self.linear(x)
        x = x.view(-1, self.config.label_size)
        return x