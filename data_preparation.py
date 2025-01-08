import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Fix path handling"""
    images = [img/255.0 for img in images]

# Refactor: 2025-01-08 - 2/3
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    return [np.fliplr(img) for img in samples]
