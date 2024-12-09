from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Update evaluation metrics"""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Fix: 2024-12-09 - 2/3
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_test, y_test):
    """Fix threshold"""
    preds = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, preds), "f1": f1_score(y_test, preds)}
