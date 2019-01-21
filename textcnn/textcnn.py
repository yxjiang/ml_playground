import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.autograd as autograd

from model import TextCNN
from dataset import TextDataset

class Config(object):
    def __init__(self, word_embedding_dim=64, word_num=20000,
                 epoch=2, sentence_max_size=40, cuda=False,
                 label_size=2, learning_rate=0.01, batch_size=1,
                 out_channel=100):
        self.word_embedding_dim = word_embedding_dim
        self.word_num = word_num
        self.epoch = epoch                                           
        self.sentence_max_size = sentence_max_size                   
        self.label_size = label_size                                  
        self.lr = learning_rate
        self.batch_size = batch_size
        self.out_channel=out_channel
        self.cuda = cuda


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_size', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

# Create the configuration
config = Config(sentence_max_size=50,
                batch_size=args.batch_size,
                word_num=11000,
                label_size=args.label_size,
                learning_rate=args.lr,
                cuda=args.gpu,
                epoch=args.epoch,
                out_channel=args.out_channel)


# Prepare the dataset
print(os.getcwd())
training_set = TextDataset(path='textcnn/data/train')
training_iter = data.DataLoader(dataset=training_set, batch_size=config.batch_size, num_workers=1)

model = TextCNN(config)
embeds = nn.Embedding(config.word_num, config.word_embedding_dim)

if torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr)

# Training
def train():
    count = 0
    loss_sum = 0
    for epoch in range(config.epoch):
        for text, label in training_iter:
            word_matrix = torch.tensor([training_set.word_to_id[w] for w in text[0].split(" ")], dtype=torch.long).cuda()

            input_data = embeds(word_matrix)
            out = model(input_data)
            loss = criterion(out, label.cuda())

            loss_sum += loss.data[0]
            count += 1

            if count % 100 == 0:
                print("Epoch %d, loss: %.5f." % (epoch, loss_sum / 100))
                loss_sum = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Save the model after each epoch
        model.save("checkpoints/epoch{}.ckpt".format(epoch))



if __name__ == "__main__":
    print("TextCNN training...")
    train()
