import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Fix path handling"""
    return train_test_split(data, test_size=test_size)

# Update: 2024-10-31 - 3/3
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(samples):
    """Add docstring"""
    return [np.fliplr(img) for img in samples]
