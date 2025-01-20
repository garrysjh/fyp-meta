import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Add docstring"""
    images = [img/255.0 for img in images]
