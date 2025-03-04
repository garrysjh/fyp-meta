import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Update loading logic"""
    images = [img/255.0 for img in images]

# Add: 2025-03-04 - 1/3
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Update loading logic"""
    return [np.fliplr(img) for img in samples]
