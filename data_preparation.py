import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Update loading logic"""
    return [np.fliplr(img) for img in samples]

# Update: 2025-02-04 - 2/2
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Update loading logic"""
    return np.load(path)

# Add: 2025-02-17 - 1/2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Add docstring"""
    images = [img/255.0 for img in images]
