# SG-1_DC_Project

<h2>Problem Statement</h2><br>
To develop a recognition system that classifies American Sign Language (ASL) gestures from video sequences. A hybrid CNN-LSTM architecture. A trained CNN extracts spatial hand features from individual frames, which are then fed into a Deep LSTM to model the movement for final classification.

<h2>Dataset<br></h2>
Dataset : https://data.mendeley.com/datasets/8fmvr9m98w/3<br>
All the datasets you will need to train the models will be available in this website. You can also refer to the .py files on that page for loading the data once its downloaded. 

<h2>Task 1: Spatial Feature Learning (Static CNN) - Deadline :  2nd March<br></h2>
Goal: Train a custom CNN model to recognize the 26 static ASL letters from individual images.
* Dataset: Use SignAlphaSet.zip.
* Action: Train a custom CNN with classifies the image into one of the 26 letters. Feel free to use any existing architectures, but code it from scratch using PyTorch. But experiment with different kernels, block sizes and other parameters. Also ensure that your model doesn’t overfit, so make sure to keep a train validation split and judge accordingly. This would be the base of the entire model, so try to get some good accuracies.
* Output: Save the weights and push that file along with your model code to your branch.

<h2>Task 2: Temporal Sequence Learning (LSTM) - Deadline :  9th March<br></h2>
Goal: Incorporate temporal knowledge with LSTMs.
* Dataset: Use ASL_dynamic.zip.
* Action: Initialize a model that processes the output from the CNN by an LSTM and classifies it using an MLP head. Load the weights into the CNN backbone and freeze them(set requires_grad=False).
* Architecture:
    1. CNN: Extracts a 512-dim vector per frame.
    2. LSTM: Processes a sequence of 16 vectors.
    3. FC Head: Classifies the final motion into gestures or letters.

<h2>Task 3: Integration & Real-Time Inference - Deadline :  16th March</h2><br>
Goal: Deploy the model to work on live video.
* Task: Create a "Sliding Window" buffer.
* Logic: As new frames come in from a webcam, maintain a queue of the last 16 frames. Every time a new frame enters, the model predicts the gesture based on that 16-frame window.
