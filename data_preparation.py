import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    images = [img/255.0 for img in images]
