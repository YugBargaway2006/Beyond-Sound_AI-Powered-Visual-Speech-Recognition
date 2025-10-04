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
from matplotlib import pyplot as plt
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



'''
  Importing required models from PyTorch for the neural network creation.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler # Base class for custom schedulers



'''
  Import the required local utility function dependencies.

'''

from loader_functions import load_video
from loader_functions import load_alignments
from loader_functions import load_data



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
  PyTorch Dataset class to fetch items and other operation requirements.

'''

class LipNetDataset(Dataset):
    def __init__(self, data_dir='../dataset/data/s1/*.mpg'):
        self.file_paths = glob.glob(data_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        frames, alignments = load_data(path)
        return frames, alignments
    


'''
  For a complete batch of items, we need to pad the items accordingly and return processed items.

'''

def collate_fn(batch):
    frames, alignments = zip(*batch)

    frame_lengths = torch.tensor([f.shape[0] for f in frames])
    alignment_lengths = torch.tensor([len(a) for a in alignments])

    # Pad sequences
    padded_frames = pad_sequence(frames, batch_first=True, padding_value=0)
    padded_alignments = pad_sequence(alignments, batch_first=True, padding_value=0)

    # Permute to (N, C, T, H, W) for Conv3D
    padded_frames = padded_frames.permute(0, 4, 1, 2, 3)

    return padded_frames, padded_alignments, frame_lengths, alignment_lengths



'''
  Model is created as a sequential type which contains three 3D convolutional layers, followed by a time distributed
  flattening layer, two Bidirectional GRU and finally a dense layer with 28 outputs and softmax activation for classification
  problem.

'''


class LipNet(nn.Module):
    def __init__(self, vocab_size):
        super(LipNet, self).__init__()

        self.conv_block = nn.Sequential(
            # Input: (N, 3, 75, 50, 100) -> (N, C, T, H, W)
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.gru1 = nn.GRU(
            input_size=1728, # 96 * 3 * 6
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.5)

        self.gru2 = nn.GRU(
            input_size=512,  # 256 * 2 from bidirectional GRU
            hidden_size=256,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.5)

        self.fc = nn.Linear(512, vocab_size) # 256 * 2

    def forward(self, x):
        # x is (N, C, T, H, W)
        x = self.conv_block(x)

        x = x.permute(0, 2, 1, 3, 4)
        batch_size, time_steps, _, _, _ = x.size()
        x = x.reshape(batch_size, time_steps, -1)

        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)

        x = self.fc(x)

        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=2)
    


'''
  Weight initialization is one of the important techniques to get better convergence
  and save from felling into the trap of local minima

'''

def initialize_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)



'''
  Fetches a single batch, runs inference, decodes the output,
  and prints the original vs. predicted text, at the end of each epoch

'''


def produce_example(model, dataloader, num_to_char, device):
    model.eval()

    padded_frames, padded_alignments, frame_lengths, alignment_lengths = next(iter(dataloader))
    padded_frames = padded_frames.to(device)

    with torch.no_grad():
        log_probs = model(padded_frames)
        predicted_indices = torch.argmax(log_probs, dim=2)


    for i in range(padded_frames.size(0)):
        original_indices = padded_alignments[i][:alignment_lengths[i]]
        original_text = ''.join([num_to_char[idx.item()] for idx in original_indices])

        pred_indices_item = predicted_indices[:, i]

        uniqued_indices = torch.unique_consecutive(pred_indices_item)
        filtered_indices = [i for i in uniqued_indices if i != 0]

        predicted_text = ''.join([num_to_char[idx.item()] for idx in filtered_indices])

        print(f"Original:    {original_text}")
        print(f"Prediction:  {predicted_text}")
        print('~'*100)


    model.train()   # Set the model back to training mode for the next epoch




