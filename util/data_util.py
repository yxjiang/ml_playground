"""
Download the file from specified location.
"""

import requests
import os
import zipfile
import tarfile
from collections import defaultdict

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
                sub_vocab |= set(f.readlines())
        else:
            raise Exception("Suffix [%s] not supported for calculating the vocabulary." % (file_suffix))
        vocab |= sub_vocab
    word_to_id = defaultdict(int)
    for i, w in enumerate(vocab, 1):
        word_to_id[w] = i
    return word_to_id
