from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Fix threshold"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# Update: 2025-03-05 - 1/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Optimize: 2025-03-14 - 1/2
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Fix threshold"""
    preds = model.predict(X_test)
    return np.mean(preds == y_test)

# Add: 2025-03-21 - 1/3
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Add logging"""
    preds = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds)}

# Remove: 2025-03-25 - 1/3
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Fix threshold"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
