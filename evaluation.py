import torch
import numpy as np
from data_preparation import MovieLensMetaDataset
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Define a relevance threshold for binary ground truth
RELEVANCE_THRESHOLD = 4.0

def mse(pred_scores, ground_truth):
    """Calculate Mean Squared Error (MSE) for predicted and ground truth ratings."""
    pred_scores = np.array(pred_scores)
    ground_truth = np.array(ground_truth)
    return np.mean((pred_scores - ground_truth) ** 2)

def rmse(pred_scores, ground_truth):
    """Calculate Root Mean Squared Error (RMSE) for predicted and ground truth ratings."""
    return np.sqrt(mse(pred_scores, ground_truth))

def binary_relevance(ground_truth):
    """Convert ground truth ratings to binary relevance based on a threshold."""
    return (ground_truth >= RELEVANCE_THRESHOLD).astype(int)

def ndcg_at_k(pred_scores, ground_truth, k=10):
    """Calculate NDCG@k for a single user's ranked list of items."""
    if len(ground_truth) == 0:
        return 0.0

    ground_truth = binary_relevance(ground_truth)
    sorted_indices = np.argsort(-pred_scores)
    ranked_items = ground_truth[sorted_indices[:k]]

    dcg = sum((ranked_items[i] / np.log2(i + 2)) for i in range(len(ranked_items)))
    idcg = sum((1.0 / np.log2(i + 2)) for i in range(min(len(ground_truth), k)))

    # Log values to inspect calculations
    # print("DCG:", dcg, "IDCG:", idcg, "NDCG:", dcg / idcg if idcg > 0 else 0.0)
    
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(pred_scores, ground_truth, k=10):
    """Calculate Precision@k for a single user's ranked list of items."""
    if len(ground_truth) == 0:
        return 0.0

    ground_truth = binary_relevance(ground_truth)
    sorted_indices = np.argsort(-pred_scores)
    ranked_items = ground_truth[sorted_indices[:k]]
    return np.sum(ranked_items) / k

def recall_at_k(pred_scores, ground_truth, k=10):
    """Calculate Recall@k for a single user's ranked list of items."""
    if len(ground_truth) == 0:
        return 0.0

    ground_truth = binary_relevance(ground_truth)
    sorted_indices = np.argsort(-pred_scores)
    ranked_items = ground_truth[sorted_indices[:k]]
    return np.sum(ranked_items) / np.sum(ground_truth) if np.sum(ground_truth) > 0 else 0.0

def binary_relevance(ground_truth):
    """Convert ground truth ratings to binary relevance based on a threshold."""
    ground_truth = np.array(ground_truth)  # Ensure ground_truth is a numpy array
    return (ground_truth >= RELEVANCE_THRESHOLD).astype(int)

def evaluate_model(model, dataset, k=10):
    """Evaluate NDCG@k, Precision@k, and Recall@k for all users in the test set."""
    start_time = time.time()
    logger.info("Starting model evaluation...")

    ndcg_scores, precision_scores, recall_scores = [], [], []
    mse_scores, rmse_scores = [], []
    
    for task in dataset.create_meta_batch(len(dataset.valid_users)):
        query_x = task['query'][:, 0].long()
        query_y = task['query'][:, 1].float()  # Load as float for thresholding

        with torch.no_grad():
            query_pred = model(query_x)

        # Calculate metrics for this batch of users
        for i in range(query_x.size(0)):
            pred_scores = query_pred[i].cpu().numpy()
            true_ratings = np.array([query_y[i].cpu().numpy()])  # Ensure this is a numpy array

            ndcg_score = ndcg_at_k(pred_scores, true_ratings, k)
            precision_score = precision_at_k(pred_scores, true_ratings, k)
            recall_score = recall_at_k(pred_scores, true_ratings, k)
            

            # Calculate regression metrics
            mse_score = mse(pred_scores, true_ratings)
            rmse_score = rmse(pred_scores, true_ratings)
            
            ndcg_scores.append(ndcg_score)
            precision_scores.append(precision_score)
            recall_scores.append(recall_score)
            mse_scores.append(mse_score)
            rmse_scores.append(rmse_score)
    
    end_time = time.time()
    logger.info(f"Model evaluation completed in {end_time - start_time:.2f} seconds")
    
    # Average scores across users
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    results = {
        'NDCG@k': avg_ndcg,
        'Precision@k': avg_precision,
        'Recall@k': avg_recall,
        'MSE': np.mean(mse_scores) if mse_scores else 0.0,
        'RMSE': np.mean(rmse_scores) if rmse_scores else 0.0
    }
    print("Evaluation Results:", results)
    
    return results

def test_metrics():
    pred_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    ground_truth = np.array([5, 3, 4, 1, 2])  # Ratings to be converted to binary relevance

    ground_truth_bin = binary_relevance(ground_truth)
    print("Binary relevance:", ground_truth_bin)

    k = 3
    ndcg = ndcg_at_k(pred_scores, ground_truth, k)
    recall = recall_at_k(pred_scores, ground_truth, k)

    print("Test metrics: ")
    print(f"NDCG@{k}:", ndcg)
    print(f"Recall@{k}:", recall)


# Example usage
if __name__ == "__main__":
    # Test the functions before execution
    test_metrics()

    # Initialize dataset
    dataset = MovieLensMetaDataset(
        data_path='data/ml-32m/ratings.csv',
        n_support=10,
        n_query=5
    )
    
    # Load trained model
    from maml import MAMLRecommender
    model = MAMLRecommender(n_items=dataset.n_items)
    model.load_state_dict(torch.load('maml_recommender.pth', weights_only=True))
    
    # Evaluate the model
    results = evaluate_model(model, dataset, k=10)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")