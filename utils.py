import cv2
import numpy as np

def process_image(frame, shape=(84, 84)):
    # Convert to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Crop the image to focus on the playing area (Breakout typically crops a bit from top and bottom)
    # Adjust cropping if necessary; here's a common choice:
    frame = frame[30:195, :]  # crop vertical from 30 to 195 pixels, keep full width
    
    # Resize the cropped frame
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    
    # Normalize pixel values (optional, your model already divides by 255)
    # frame = frame.astype(np.float32) / 255.0
    
    # Reshape to (1, 84, 84) for channel dimension
    frame = frame.reshape(1, *shape)
    
    return frame.astype(np.uint8)  # ensure uint8 type
