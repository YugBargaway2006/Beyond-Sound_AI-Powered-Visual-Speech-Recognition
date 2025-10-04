'''
  Import the required dependencies.

'''

import os
import gdown



'''
  Download a part of dataset for representation purpose of working of the pipeline.
  The original dataset contains multiple speakers (34), but for the ease of training
  and computational limitations, we are considering only 2 speakers, consisting of 1000 videos,
  out of which 900 is in training and 100 is in testing partition.

  Download link of the dataset is provided as a google drive link. If in any case it fails,
  put an issue in the github repo, it will be fixed as soon as possible

'''

url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output_file = 'data.zip'

gdown.download(url, output_file, quiet=False)
gdown.extractall('data.zip')

