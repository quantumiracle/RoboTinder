import gdown
import os

# # url = 'https://drive.google.com/drive/folders/1nzMJ-d4kE6Awu1ewCbBOjSbdBJeJE4RS?usp=share_link'
# url = 'https://drive.google.com/drive/folders/1WWOzM9T4HUvuB6gUYPqIEOb_aZwUULNA?usp=share_link'
# output = './'
# id = url.split('/')[-1]
# # directly download is denied sometimes (not stable)
# # gdown.download_folder(url, quiet=False, use_cookies=False)
# # gdown.download(id=id, output=output, quiet=False, use_cookies=False)

# os.system(f"gdown --id {id} -O {output} --folder --no-cookies --remaining-ok")

import zipfile
from os import listdir
from os.path import isfile, join, isdir

path_to_zip_file = './processed_zip/'
zip_files = [join(path_to_zip_file, f) for f in listdir(path_to_zip_file)]
for f in zip_files:
    if f.endswith(".zip"):
        directory_to_extract_to = path_to_zip_file # extracted file itself contains a folder
        print(f'extract data {f} to {directory_to_extract_to}')
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        os.remove(f)
