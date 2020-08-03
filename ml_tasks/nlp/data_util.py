"""
Download the file from specified location.
"""

import requests
import os
import zipfile
import tarfile
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


def download_data(url, dest_dir="/tmp/data"):
    """
    Download the contents from specified url to the destination folder.
    """
    filename = url[url.rindex("/") + 1:]
    file_format = filename[filename.index(".") + 1:]
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        print("Destination folder [%s] doest not exist, create it." % (dest_dir))
    else:
        print("Destination folder [%s] exists." % (dest_dir))

    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        print("Download data from [%s] to [%s]..." % (url, filepath))
        with open(filepath, "wb") as f:
            f.write(requests.get(url).content)
        print("Data downloaded.")
    else:
        print("Target file [%s] exists, skip downloading." % (filename))

    folder_name = filename[:filename.index(".")]
    if not os.path.exists(os.path.join(dest_dir, folder_name)):
        if file_format == 'zip':
            with zipfile.ZipFile.open(filepath, "r") as z:
                print("Start to extract [%s] to [%s]..." % (filepath, dest_dir))
                z.extractall(path=dest_dir)
                print("File extracted")
        elif file_format == 'tgz' or file_format == 'tar.gz':
            with tarfile.open(filepath, "r:gz") as t:
                print("Start to extract [%s] to [%s]..." % (filepath, dest_dir))
                t.extractall(path=dest_dir)
                print("File extracted")
        else:
            print("Unsupported compression format for [%s]" % (filename))
    else:
        print("Dataset [%s] already extracted, skip extracting." % (folder_name))

            
def get_vocabulary(folder_path, file_suffix, check_interval=50000):
    """
    Get the word to id from the vocabulary of the text corpus.
    """
    print("Processing vocabulary from [%s]." % (folder_path))
    vocab = set()
    for filename in os.listdir(folder_path):
        if not filename.endswith(file_suffix):
            continue
        filepath = os.path.join(folder_path, filename)
        sub_vocab = set()
        if file_suffix == "csv":
            data_frame = pd.read_csv(filepath)
            for i, row in data_frame.iterrows():
                if i % check_interval == 0:
                    print("Processed %d rows" % (i))
                sub_vocab |= set(row[2].strip().split(" "))
        elif file_suffix == "txt":
            with open(filepath, "r") as f:
                for line in f.readline():
                    sub_vocab |= set(line.strip().split(" "))
        elif file_suffix == "vocab":
            with open(filepath, "r") as f:
                sub_vocab |= set([w.strip() for w in f.readlines()])
        else:
            raise Exception("Suffix [%s] not supported for calculating the vocabulary." % (file_suffix))
        vocab |= sub_vocab
    word_to_id = defaultdict(lambda: 0)
    words = ['N/A']
    for i, w in enumerate(vocab, 1):  # 0 is for unknown
        word_to_id[w.lower()] = i
        words.append(w.lower())
    return word_to_id, words

# transform, dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TruncateTransform:
    """
    Truncate all sentences to the maximal allowed length.
    """
    def __init__(self, config):
        self.config = config
    
    def __call__(self, input):
        if len(input["words"]) >= self.config.sentence_max_length:
            input["words"] = input["words"][:self.config.sentence_max_length]
        return input


class WordsToIdsTransform:
    """
    Convert the list of words to embeddings.
    """
    def __init__(self, config, word_to_id):
        self.config = config
        self.word_to_id = word_to_id
    
    def __call__(self, input):
        input["word_ids"] = [self.word_to_id[w.lower()] for w in input["words"]]
        return input

class PadTransform:
    """
    Pad all sentences to the equal length.
    """
    def __init__(self, config):
        self.config = config
    
    def __call__(self, input):
        if len(input["word_ids"]) < self.config.sentence_max_length:
            input["word_ids"].extend([0] * (self.config.sentence_max_length - len(input["word_ids"])))
        return input

def pad_collate(batch):
    _, word_to_id_batch, label_batch = zip(*batch)
    x_lens = [len(word_to_id) for word_to_id in word_to_id_batch]
    y_lens = [1] * len(label_batch)
    word_to_id_pad = pad_sequence(word_to_id_batch, batch_first=True, padding_value=0)
    label_pad = pad_sequence(label_batch, batch_first=True, padding_value=0).squeeze(1)
    return word_to_id_pad, label_pad, x_lens, y_lens



class MovieReviewDataset(Dataset):
    def __init__(self, config, pos_data_folder, neg_data_folder, word_to_id, transform):
        self.config = config
        self.word_to_id = word_to_id
        self.data = []
        # read all data into memory
        for filename in os.listdir(pos_data_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(pos_data_folder, filename), "r") as f:
                    self.data.append((f.readline(), 1))

        for filename in os.listdir(neg_data_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(neg_data_folder, filename), "r") as f:
                    self.data.append((f.readline(), 0))

        self.transform = transform
    
    def __getitem__(self, idx):
        words = [w.strip() for w in self.data[idx][0].strip().split(" ")]
        label = self.data[idx][1]
        input = self.transform({"words": words, "label": label})
        return input["words"], torch.LongTensor(input["word_ids"]), torch.LongTensor([input["label"]])
        

    def __len__(self):
        return len(self.data)
