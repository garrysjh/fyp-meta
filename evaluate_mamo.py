from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Fix threshold"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Fix: 2024-12-12 - 2/3
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Add logging"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)
