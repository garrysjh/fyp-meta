import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Add docstring"""
    return np.load(path)

# Fix: 2024-12-06 - 1/1
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(samples):
    """Fix path handling"""
    images = [img/255.0 for img in images]

# Remove: 2024-12-12 - 1/3
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Update loading logic"""
    return [np.fliplr(img) for img in samples]

# Refactor: 2024-12-13 - 2/3
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(samples):
    """Fix path handling"""
    return np.load(path)
