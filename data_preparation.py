import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Fix path handling"""
    return [np.fliplr(img) for img in samples]

# Optimize: 2024-11-06 - 1/2
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Update loading logic"""
    images = [img/255.0 for img in images]
