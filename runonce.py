import gdown
import zipfile
import sys


train_zip = "train.zip"
val_zip = "val.zip"
test_zip = "test_public.zip"


# Check if the argument passed says "download"
if "download" in sys.argv:
    download_url = f"https://drive.google.com/uc?id=1YkGwaxBKNiYL2nq--cB6WMmYGzRmRKVr"
    gdown.download(download_url, train_zip, quiet=False)  # Downloads the file to your drive

    download_url = "https://drive.google.com/uc?id=1wtmT_vH9mMUNOwrNOMFP6WFw6e8rbOdu"
    gdown.download(download_url, val_zip, quiet=False)

    download_url = "https://drive.google.com/uc?id=1G9xGE7s-Ikvvc2-LZTUyuzhWAlNdLTLV"
    gdown.download(download_url, test_zip, quiet=False)

with zipfile.ZipFile(train_zip, 'r') as zip_ref:  # Extracts the downloaded zip file
    zip_ref.extractall(".")

with zipfile.ZipFile(val_zip, 'r') as zip_ref:
    zip_ref.extractall(".")

with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(".")