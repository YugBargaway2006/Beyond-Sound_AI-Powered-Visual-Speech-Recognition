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
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn



'''
  Import the required local utility function dependencies.

'''

from loader_functions import load_video
from loader_functions import load_alignments
from loader_functions import load_data
from lipnet_nn_model import collate_fn
from lipnet_nn_model import LipNetDataset
from lipnet_nn_model import LipNet
from lipnet_nn_model import initialize_weights
from lipnet_nn_model import produce_example



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
  Declaring Vocabulary

'''

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz "]



'''
  Creating dictionary of vocab to map to numerical values.

'''

vocab_full = [''] + vocab   # Adding the out-of-vocabulary token

char_to_num = {char: i for i, char in enumerate(vocab_full)}

num_to_char = {i: char for i, char in enumerate(vocab_full)}

print(
    f"The vocabulary is: {vocab_full} "
    f"(size ={len(vocab_full)})"
)



'''
    Testing if conversion working properly

'''

name = ['y', 'u', 'g', ' ', 'b', 'a', 'r', 'g', 'a', 'w', 'a', 'y']
idx = [char_to_num[char] for char in name]
idx_tsr = torch.tensor(idx)
print(idx_tsr)

ch = [num_to_char[i] for i in [25, 21,  7, 27,  2,  1, 18,  7,  1, 23,  1, 25]]
print(ch)



'''
    Testing if file handling working properly

'''

test_path = '../dataset/data/s1/bbal6n.mpg'
file_name = os.path.splitext(os.path.basename(test_path))[0]

print(file_name)

frames, alignments = load_data(test_path)
plt.imshow(frames.numpy()[20])

print(alignments)

text = ''.join([num_to_char[i.item()] for i in alignments])
print(text)




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

first_video_tensor = val[0][0]
frame_tensor = first_video_tensor[:, 36, :, :]   # 36th frame from the first video
frame_to_plot = frame_tensor.permute(1, 2, 0).numpy()

plt.imshow(frame_to_plot)

first_alignment_tensor = val[1][0]
text = ''.join([num_to_char[i.item()] for i in first_alignment_tensor])

print(text)



'''
    Model Build up and training

'''

# The vocabulary size is 27 characters + 1 blank token for CTC
vocab_size = 28

model = LipNet(vocab_size=vocab_size).to(device)
model.apply(initialize_weights)




'''
  Initiating the Adam optimizer and learning rate scheduler and CTC Loss.

'''

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Halve the LR instead of dividing by 5
    patience=5,     # Wait 5 epochs instead of 10
)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)



'''
    Model Training

'''

'''
  Complete training pipeline with callbacks, optimizer, learning rate scheduler
  and example prediction.

'''


epochs = 100
start_epoch = 0
best_loss = float('inf')
save_path = os.path.join('..', 'checkpoints', 'checkpoint.weights.pt')
os.makedirs('models', exist_ok=True)

# Check if a checkpoint file exists to resume from
if os.path.exists(save_path):
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(save_path))
    start_epoch = 74   # Enter the checkpoint epoch
    print(f"Resuming training from epoch {start_epoch + 1}")


# Main Training Loop
for epoch in range(start_epoch, epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")

    model.train()
    running_loss = 0.0
    for batch_idx, (frames, alignments, frame_lengths, alignment_lengths) in enumerate(train_loader):
        frames = frames.to(device)
        alignments = alignments.to(device)

        optimizer.zero_grad()

        log_probs = model(frames)

        loss = criterion(log_probs, alignments, frame_lengths, alignment_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0: # Print progress every 10 batches
            print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Training Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    print(f"End of Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for frames, alignments, frame_lengths, alignment_lengths in test_loader:
            frames = frames.to(device)
            alignments = alignments.to(device)

            log_probs = model(frames)
            val_loss = criterion(log_probs, alignments, frame_lengths, alignment_lengths)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(test_loader)
    print(f"End of Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")

    # Callbacks Logic
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"-> Checkpoint saved to {save_path}")

    scheduler.step(avg_val_loss)

    print("\n--- Generating example prediction ---")
    produce_example(model, test_loader, num_to_char, device)



'''
    Single Batch Testing Overfit

'''

# # --- Complete Single-Batch Overfitting Test ---
# print("Fetching a single batch to overfit...")
# frames, alignments, frame_lengths, alignment_lengths = next(iter(train_loader))
# frames, alignments = frames.to(device), alignments.to(device)

# print("Starting single-batch overfitting test...")
# model.train()
# for i in range(400):
#     optimizer.zero_grad()
#     log_probs = model(frames)
#     loss = criterion(log_probs, alignments, frame_lengths, alignment_lengths)
#     loss.backward()
#     optimizer.step()

#     if (i + 1) % 20 == 0:
#         print(f"  Iteration {i+1}, Loss: {loss.item():.4f}")

#         # --- Decode and Print Prediction ---
#         predicted_indices = torch.argmax(log_probs.detach(), dim=2)

#         # Get the first item from the batch to display
#         pred_indices_item = predicted_indices[:, 0]
#         uniqued_indices = torch.unique_consecutive(pred_indices_item)
#         filtered_indices = [idx.item() for idx in uniqued_indices if idx.item() != 0]
#         predicted_text = ''.join([num_to_char[idx] for idx in filtered_indices])

#         # Get the original text for comparison
#         original_indices = alignments[0][:alignment_lengths[0]]
#         original_text = ''.join([num_to_char[idx.item()] for idx in original_indices])

#         print(f"    Original:   '{original_text}'")
#         print(f"    Prediction: '{predicted_text}'")