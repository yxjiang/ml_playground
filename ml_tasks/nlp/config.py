from util import config
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TextCNNConfig(config.Config):
    def __init__(
            self, criteria=nn.CrossEntropyLoss, optimizer=optim.Adam, lr=0.00003, epochs=1000, 
            batch_size=1024, num_classes=2, sentence_max_length=20, word_embedding_length=32, 
            activation=F.relu, dropout=0.1, conv_layer_sizes=[3,4,5], **kwargs
        ):
        super(TextCNNConfig, self).__init__(criteria, optimizer, lr, epochs, batch_size, kwargs)
        self.num_classes = num_classes
        self.sentence_max_length = sentence_max_length
        self.word_embedding_length = word_embedding_length
        self.activation = activation
        self.dropout = dropout
        self.conv_layer_sizes = conv_layer_sizes

    def __str__(self):
        return "sentence_max_len_%d-embedding-%d-lr-%.8f-batch_size-%d-dropout-%.2f-conv_layers-%s" % (
            self.sentence_max_length, self.word_embedding_length, self.lr, self.batch_size, self.dropout, 
            "|".join([str(s) for s in self.conv_layer_sizes])
        )

class RNNConfig(config.Config):
    def __init__(
            self, rnn_type=nn.RNN, criteria=nn.CrossEntropyLoss, optimizer=optim.Adam, lr=0.00003, epochs=1000, 
            batch_size=1024, num_classes=2, sentence_max_length=20, word_embedding_length=32, 
            num_layers=1, activation=F.relu, dropout=0.1, **kwargs
    ):
        super(RNNConfig, self).__init__(criteria, optimizer, lr, epochs, batch_size, **kwargs)
        self.rnn_type = rnn_type
        self.num_classes = num_classes
        self.sentence_max_length = sentence_max_length
        self.word_embedding_length = word_embedding_length
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout

    def __str__(self):
        rnn_name = None
        if self.rnn_type == nn.RNN:
            rnn_name = "RNN"
        elif self.rnn_type == nn.LSTM:
            rnn_name = "LSTM"
        elif self.rnn_type == nn.GRU:
            rnn_name = "GRU"
        return "sentence_max_len_%d-embedding-%d-model-%s-layers-%d-lr-%.8f-batch_size-%d-dropout-%.2f" % (
            self.sentence_max_length, self.word_embedding_length, self.rnn_type, self.num_layers, 
            self.lr, self.batch_size, self.dropout, 
        )