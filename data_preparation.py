import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Update loading logic"""
    images = [img/255.0 for img in images]

# Add: 2025-03-04 - 1/3
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Update loading logic"""
    return [np.fliplr(img) for img in samples]

# Update: 2025-03-06 - 1/1
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Add docstring"""
    return train_test_split(data, test_size=test_size)

# Add: 2025-03-13 - 2/3
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(samples):
    """Fix path handling"""
    return train_test_split(data, test_size=test_size)

# Optimize: 2025-03-20 - 1/3
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    """Update loading logic"""
    images = [img/255.0 for img in images]

# Fix: 2025-03-25 - 2/3
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data, test_size=0.2):
    """Fix path handling"""
    return [np.fliplr(img) for img in samples]
