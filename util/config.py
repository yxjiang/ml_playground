# configs
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class Config:
    def __init__(self, criteria=nn.CrossEntropyLoss, optimizer=optim.Adam, lr=0.001, epochs=500, batch_size=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criteria = criteria
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        