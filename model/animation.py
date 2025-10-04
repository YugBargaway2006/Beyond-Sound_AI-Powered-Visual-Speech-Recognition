'''
  Import the required dependencies.

'''

import os
import cv2
import torch
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
import imageio
from torchvision import transforms
import gdown
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



'''
  Import the required local utility function dependencies.

'''

from loader_functions import load_video
from loader_functions import load_alignments
from loader_functions import load_data
from lipnet_nn_model import collate_fn
from lipnet_nn_model import LipNetDataset



'''
  Check if the libraries are properly imported.

'''


print("OpenCV version: ", cv2.__version__)
print("NumPy version: ", np.__version__)
print("ImageIO version: ", imageio.__version__)



'''
  Check if GPU is available in the system. We will shift the model to GPU for faster training.

'''

gpu_aval = torch.cuda.is_available()
print(f"GPU available: {gpu_aval}")

if gpu_aval:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



'''
  Creating train and test dataloader to facililate the training and testing process

'''

full_dataset = LipNetDataset()

train_size = 900
test_size = 100
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2
)

print(f"Total samples: {len(full_dataset)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

len(test_loader)



'''
  Creating an iterator from train_loader dataloader

'''

data_iterator = iter(train_loader)

frames, alignments, _, _ = next(data_iterator)

print(f"Frames tensor shape: {frames.shape}")
print(f"Frames tensor dtype: {frames.dtype}")
print(f"Alignments tensor shape: {alignments.shape}")
print(f"Alignments tensor dtype: {alignments.dtype}")

len(frames)

sample_iterator = iter(train_loader)

val = next(sample_iterator)

print(val[0])


video_tensor = val[0][0]   # Took first video frames out from the batch

# Permute the tensor from (C, T, H, W) to (T, H, W, C) for imageio
frames_to_save = video_tensor.permute(1, 2, 3, 0)

frames_for_gif = frames_to_save.to(torch.uint8).numpy()

imageio.mimsave('./animation.gif', frames_for_gif, fps=10)