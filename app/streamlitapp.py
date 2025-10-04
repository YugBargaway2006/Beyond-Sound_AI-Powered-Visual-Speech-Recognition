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
import streamlit as st



'''
  Import the required local utility function dependencies.

'''

from model.loader_functions import load_video
from model.loader_functions import load_alignments
from model.loader_functions import load_data
from model.lipnet_nn_model import collate_fn
from model.lipnet_nn_model import LipNetDataset
from model.lipnet_nn_model import LipNet
from model.lipnet_nn_model import initialize_weights
from model.lipnet_nn_model import produce_example
from model.train_nn import num_to_char
from predictor.testing import predict_video



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
    Streamlit app architecture and python backend

'''

st.set_page_config(layout='wide')

with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Beyond-Sound_AI-Powered-Visual-Speech-Recognition')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Beyond-Sound_AI-Powered-Visual-Speech-Recognition') 

# list of options or videos 
options = os.listdir(os.path.join('..', 'dataset', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)


col1, col2 = st.columns(2)

if options: 

    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'dataset', 'data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')

        model_checkpoint_path = 'models/checkpoint.weights.pt' 
        real_text, predicted_text = predict_video(model_checkpoint_path, file_path, device, num_to_char)
        
        st.image('animation.gif', width=400) 

        st.info('This is the real transcipt of the person.')
        st.text(real_text)

        st.info('This is the output of the machine learning model.')
        
        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        st.text(predicted_text)
