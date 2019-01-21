# Prepare the dataset
import os
from torch.utils import data 
class TextDataset(data.Dataset):
    def __init__(self, path):
        self.train_set = []
        vocab = set()
        with open(path) as f:
            for line in f.readlines():
                text, label = line.split(",")
                vocab |= set(text.split(" "))
                self.train_set.append((text, label.strip()))  # lists of text, label
        self.word_to_id = {word: i for i, word in enumerate(vocab)}

    def __getitem__(self, index):
        return self.train_set[index][0], self.train_set[index][1]

    def __len__(self):
        return len(self.train_set)