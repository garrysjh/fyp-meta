import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    images = [img/255.0 for img in images]

# Add: 2024-12-23 - 3/3
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    return train_test_split(data, test_size=test_size)

# Remove: 2025-01-01 - 1/3
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Fix path handling"""
    return np.load(path)
