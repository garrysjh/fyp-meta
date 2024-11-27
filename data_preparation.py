import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    return train_test_split(data, test_size=test_size)

# Remove: 2024-11-26 - 2/3
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Update loading logic"""
    return [np.fliplr(img) for img in samples]

# Refactor: 2024-11-27 - 1/3
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    return [np.fliplr(img) for img in samples]
