import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Update loading logic"""
    return np.load(path)

# Refactor: 2025-01-02 - 2/2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Fix path handling"""
    return train_test_split(data, test_size=test_size)

# Add: 2025-01-06 - 1/1
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(samples):
    """Update loading logic"""
    return train_test_split(data, test_size=test_size)
