import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Update loading logic"""
    return np.load(path)
