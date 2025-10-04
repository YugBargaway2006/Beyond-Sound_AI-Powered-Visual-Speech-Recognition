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



'''
  Check if the libraries are properly imported.

'''

print("OpenCV version: ", cv2.__version__)
print("NumPy version: ", np.__version__)
print("ImageIO version: ", imageio.__version__)



'''
  Creating the custom video loader to preprocess the video frames.
  It loads a video, extracts frames, converts to grayscale, crops the lip region,
  and standardizes the pixel values.

'''

def load_video(path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
       
        # Placeholder to apply DLip Detector (Have to add it here)

        mouth_crop = frame[190:240, 100:200] # Results in 50x100
        mouth_crop_resized = cv2.resize(mouth_crop, (100, 50))

        frames.append(mouth_crop_resized)
    cap.release()

    frames_np = np.stack(frames)
    
    frames_np = frames_np[..., ::-1].copy() # BGR -> RGB
    frames_tensor = torch.from_numpy(frames_np).float()
    
    mean = frames_tensor.mean(dim=(0, 1, 2))
    std = frames_tensor.std(dim=(0, 1, 2))
    standardized_frames = (frames_tensor - mean) / (std + 1e-6)

    # Return shape (T, H, W, C)
    return standardized_frames



'''
  Load transcripts of video and tokenize it accordingly
  It reads an alignment file, extracts the words, converts them to a sequence
  of character indices, and returns them as a tensor.

'''

def load_alignments(path: str) -> torch.Tensor:
    with open(path, 'r') as f:
        lines = f.readlines()

    words = [line.split()[2] for line in lines if line.split()[2] != 'sil']   # Ignore silence
    text = ' '.join(words)
    tokens = [char_to_num[char] for char in text]

    return torch.tensor(tokens, dtype=torch.long)



'''
  Complete data loader function, which loads the video and maps it with its corresponding transcript.
  It takes a file path to an mpg video, constructs the corresponding alignment path,
  and loads both the video frames and alignment tokens.

'''

def load_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = os.path.basename(path)   # file name
    file_name, _ = os.path.splitext(bs)

    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


