import urllib.request
import zipfile
import os

url = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
zip_path = "dataset.zip"

print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("data")

os.remove(zip_path)
print("Done! Check the 'data' folder.")