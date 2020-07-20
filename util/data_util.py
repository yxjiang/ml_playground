"""
Download the file from specified location.
"""

import requests
import os
import zipfile
import tarfile

def download_data(url, dest_dir="/tmp/data"):
    """
    Download the contents from specified url to the destination folder.
    """
    filename = url[url.rindex("/") + 1:]
    file_format = url[url.rindex(".") + 1:]
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        print("Destination folder [%s] doest not exist, create it." % (dest_dir))
    else:
        print("Destination folder exists.")

    filepath = os.path.join(dest_dir, filename)
    if not os.path.exists(filepath):
        print("Download data from [%s] to [%s]..." % (url, filepath))
        with open(filepath, "wb") as f:
            f.write(requests.get(url).content)
        print("Data downloaded.")
    else:
        print("Target file exists, skip downloading.")

    folder_name = filename[:filename.rindex(".")]
    if not os.path.exists(os.path.join(dest_dir, folder_name)):
        if file_format == 'zip':
            with zipfile.ZipFile.open(filepath, "r") as z:
                print("Start to extract [%s] to [%s]." % (filepath, dest_dir))
                z.extractall(path=dest_dir)
                print("File extracted")
        elif file_format == 'tgz':
            with tarfile.open(filepath, "r:gz") as t:
                print("Start to extract [%s] to [%s]." % (filepath, dest_dir))
                t.extractall(path=dest_dir)
                print("File extracted")
        else:
            print("Unsupported compression format for [%s]" % (filename))
    else:
        print("Data extracted, skip extracting.")

            

