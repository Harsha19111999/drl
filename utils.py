import cv2
import numpy as np
from config import *

def process_image(frame, shape=(84, 84)):
    # Convert to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Crop the image to focus on the playing area (Breakout typically crops a bit from top and bottom)
    # Adjust cropping if necessary; here's a common choice:
    # (NUM_ENV, HEIGHT, WIDTH) CHW
    grayscale_frames = []
    for i in range(frame.shape[0]):
        grayscale_frames.append(cv2.cvtColor(frame[i, :, :, :], cv2.COLOR_RGB2GRAY))
    frame = np.array(grayscale_frames)
    frame = frame[:, 30:195, :]  # crop vertical from 30 to 195 pixels, keep full width
    frame = np.transpose(frame, (1, 2, 0)) # Open CV expected in HWC 
    # Resize the cropped frame
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = np.transpose(frame, (2, 0, 1))
    # Normalize pixel values (optional, your model already divides by 255)
    # frame = frame.astype(np.float32) / 255.0
    
    # Reshape to (NUM_ENV, 1, 84, 84) for channel dimension
    frame = frame.reshape((NUM_ENVS, 1, *shape))    
    return frame.astype(np.uint8)  # ensure uint8 type
