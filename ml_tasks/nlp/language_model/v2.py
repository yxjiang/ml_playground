import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Any, Callable

# hyperparameters
batch_size = 32
max_iters = 10000
eval_interval = 200
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_heads = 4
sequence_length = 32 



dim_embedding = 384
num_stacks = 6
dropout = 0.2


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


# Enable each position to know the relative affinity of the previous positions.
# Single head self-attention can reduce loss from 2.55 to 2.45 compare with no self-attention.
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, sequence_length: int, dim_embedding: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(dim_embedding, head_size, bias=False)
        self.query = nn.Linear(dim_embedding, head_size, bias=False)
        self.value = nn.Linear(dim_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.key(x)  # (B, T, C)
        v = self.key(x)  # (B, T, C)
        #  compute attention scores ("affinities")
        weight = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[: T, : T] == 0, float('-inf'))  # (B, T, T)
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)
        # perform the weighted aggregation of the values
        out = weight @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# Enable a position to learn the affinities from the previous positions in multiple subspaces.
# Reduce the loss from 2.45 to 2.3 from single head self-attention.
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads: int, head_size: int, sequence_length: int, dim_embedding: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size=head_size, sequence_length=sequence_length, dim_embedding=dim_embedding, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(dim_embedding, dim_embedding)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate along the channel dimension
        out = self.projection(out)
        out = self.dropout(out)
        return out


# Enable to learn the information gathered from the previous positions.
# Reduce the loss from 2.3 to 2.2. Reduce more with residual.
class FeedForward(nn.Module):
    """ a simple feed forward layer followed by a non-linearity """
    def __init__(self, dim_embedding: int, dropout: float):
        super().__init__()
        # Feed forward layer is 4x wider of the embedding dimension
        self.net = nn.Sequential(
            nn.Linear(dim_embedding, 4 * dim_embedding),
            nn.ReLU(),
            nn.Linear(4 * dim_embedding, dim_embedding),  # projection,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm1d:
    """ batch norm 1d, normalize every row to 0 mean and 1 variance """
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(dim=1, keepdim=True)
        xvar = x.var(dim=1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# Stack the self-attention.
# Reduce the loss from 2.2 to 2.08 with residule.
# Reduce the loss from 2.08 to  with layer norm.
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, dim_embedding: int, num_heads: int, sequence_length: int, dropout: float):
        super().__init__()
        head_size = dim_embedding // num_heads
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads, head_size=head_size, sequence_length=sequence_length, dim_embedding=dim_embedding, dropout=dropout)
        self.feed_forward_layer = FeedForward(dim_embedding=dim_embedding, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)
    
    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward_layer(self.layer_norm2(x))
        return x


class BigramLM(nn.Module):
    
    def __init__(self, vocab_size: int, num_heads: int, sequence_length: int, dim_embedding: int, num_stacks: int, dropout: float):
        super().__init__()
        print(f"vocab_size: {vocab_size}, head_size: {num_heads}, sequence_length: {sequence_length}, dim_embedding: {dim_embedding}")
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, dim_embedding)
        # get the embedding of the positions
        self.position_embedding_table = nn.Embedding(sequence_length, dim_embedding)
        # note the head size here is the dim_embedding
        self.blocks = nn.Sequential(*[
            Block(dim_embedding=dim_embedding, num_heads=num_heads, sequence_length=sequence_length, dropout=dropout)
            for _ in range(num_stacks)
        ])
        self.layer_norm_layer = nn.LayerNorm(dim_embedding)
        self.lm_head = nn.Linear(dim_embedding, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        # idx and targets are both with size (B, T)
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C), where C = dim_embedding
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_embeddings + pos_embeddings  # (B, T, C), pos embed will be broadcasted
        x = self.blocks(x)  # (B, T, C)
        x = self.layer_norm_layer(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # reshape as cross_entropy receive channel as the second dimension.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, sequence_length: int) -> torch.Tensor:
        """Generate the next character."""
        # idx is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # crop the idx to the last sequence_length tokens
            idx_crop = idx[:, -sequence_length : ]
            # get the predictions
            logits, loss = self(idx_crop)
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
    

def generate_sequence(
    prompt: str,
    model: nn.Module,
    encode_fun: Callable[[str], List[int]],
    decode_fun: Callable[[List[int]], str],
    sequence_length: int,
):
    # generate input from the prompt
    arr = np.array([encode_fun(prompt)])
    context = torch.as_tensor(arr, dtype=torch.long, device=device)
    # context = torch.zeros((1, 2), dtype=torch.long, device=device)
    # generate output from the model
    generated_tokens = model.generate(context, max_new_tokens=500, sequence_length=sequence_length)
    generated_text = decode_fun(generated_tokens[0].tolist())
    print(generated_text)


if __name__ == '__main__':
    should_train = False
    prompt = ' '
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            should_train = True
        elif sys.argv[1] == 'prompt':
            prompt = ' '.join(sys.argv[2:])
            if len(prompt) == 0:
                print(f"params: train/prompt [prompt words]")
                sys.exit()
            print(f"prompt: {prompt}")
    else:
        print(f"params: train/prompt [prompt words]")
        sys.exit()

    model_path = "/Users/yjiang/Downloads/gpt.model"
    # prepare data
    print('prepare data...')
    text_path = './samples/tinyshakespeare.txt'
    text = read_text_data(text_path)
    chars, vocab_size, encode_fun, decode_fun = text_data_stats(text)
    train_data, validation_data = generate_training_data(text, encode_fun)

    blm = BigramLM(
        vocab_size=vocab_size,
        num_heads=num_heads,
        sequence_length=sequence_length,
        dim_embedding=dim_embedding,
        num_stacks=num_stacks,
        dropout=dropout
    )
    if os.path.exists(model_path):
        print(f"load model from: {model_path}")
        blm.load_state_dict(torch.load(model_path))
    model = blm.to(device)
    # train model
    if should_train:
        print('train model...')
        train_blm(model)
        torch.save(model.state_dict(), model_path)
        print(f"model saved to: {model_path}")

    # try generate the sequence
    generate_sequence(prompt=prompt, model=model, encode_fun=encode_fun, decode_fun=decode_fun, sequence_length=sequence_length)
