import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Any, Callable

# hyperparameters
batch_size = 32
sequence_length = 8
max_iters = 5000
eval_interval = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


def read_text_data(path: str) -> str:
    text = None
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def text_data_stats(text: str) -> Tuple[List[str], int, Callable[[str], List[int]], Callable[[List[int]], str]]:
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    atoi = { ch: i for i, ch in enumerate(chars) }
    itoa = { i: ch for i, ch in enumerate(chars) }
    encode_fun = lambda s: [atoi[ch] for ch in s]
    decode_fun = lambda l: ''.join([itoa[i] for i in l])
    return chars, vocab_size, encode_fun, decode_fun


def generate_training_data(text: str, encode_fun) -> Tuple[torch.Tensor, torch.Tensor]:
    # separate the dataset
    data = torch.tensor(encode_fun(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    validation_data = data[n:]
    return train_data, validation_data


def get_batch(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # random sampling subsequence starting positions from the dataset
    sampled_pos = torch.randint(len(data) - sequence_length, size=(batch_size, ))
    x = torch.stack([data[i : i + sequence_length] for i in sampled_pos])
    y = torch.stack([data[i + 1 : i + sequence_length + 1] for i in sampled_pos])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int, train_data: torch.Tensor, validation_data: torch.Tensor):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(train_data if 'train' else validation_data)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLM(nn.Module):
    
    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # idx and targets are both with size (B, T)
        logits = self.token_embedding_table(idx)  # (B, T, C), where C = vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # reshape as cross_entropy receive channel as the second dimension.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate the next character."""
        # idx is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # only need the last in the sequence even the entire sequence is available. This is how the bigram model works.
            logits = logits[:, -1, :]  # (B, T, C) -> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


def train_blm(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, train_data, validation_data)
            print(f"step {iter}: train loss {losses['train']: .4f}, val loss {losses['val']: .4f}")

        # sample a batch of data
        x_batch, y_batch = get_batch(train_data)

        # calculate the loss
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # calculate the new grad
        optimizer.step()  # update the weights
    

def generate_sequence(model: nn.Module, decode_fun: Callable[[List[int]], str]):
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=500)
    generated_text = decode_fun(generated_tokens[0].tolist())
    print(generated_text)

if __name__ == '__main__':
    # prepare data
    print('prepare data...')
    text_path = './samples/tinyshakespeare.txt'
    text = read_text_data(text_path)
    chars, vocab_size, encode_fun, decode_fun = text_data_stats(text)
    train_data, validation_data = generate_training_data(text, encode_fun)
    # train model
    print('train model...')
    blm = BigramLM(vocab_size)
    model = blm.to(device)
    train_blm(model)
    # try generate the sequence
    generate_sequence(model, decode_fun)
