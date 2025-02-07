from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Add logging"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# Optimize: 2025-01-20 - 1/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Add logging"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# Remove: 2025-01-21 - 2/3
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# Refactor: 2025-01-23 - 2/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Add: 2025-01-27 - 1/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Update: 2025-01-31 - 1/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds)}

# Update: 2025-02-06 - 2/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# Fix: 2025-02-07 - 1/1
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Add logging"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)
