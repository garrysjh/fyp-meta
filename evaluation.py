import torch
import numpy as np
from data_preparation import MovieLensMetaDataset
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

RELEVANCE_THRESHOLD = 0.8  # Ratings above this are relevant

def mse(pred_scores, ground_truth):
    pred_scores, ground_truth = np.array(pred_scores), np.array(ground_truth)
    return np.mean((pred_scores - ground_truth) ** 2)

def rmse(pred_scores, ground_truth):
    return np.sqrt(mse(pred_scores, ground_truth))

def binary_relevance(ground_truth):
    """Convert ratings to binary relevance."""
    return (np.array(ground_truth) >= RELEVANCE_THRESHOLD).astype(int)

def ndcg_at_k(pred_scores, ground_truth, k=10):
    """Compute NDCG@k correctly by considering ranking order."""
    ground_truth = binary_relevance(ground_truth)
    if np.sum(ground_truth) == 0:
        return 0.0  # No relevant items

    sorted_indices = np.argsort(-pred_scores)
    valid_k = min(k, len(ground_truth))  # Ensure we don't index beyond available elements
    ranked_relevance = ground_truth[sorted_indices[:valid_k]]

    dcg = np.sum(ranked_relevance / np.log2(np.arange(2, len(ranked_relevance) + 2)))
    ideal_ranking = sorted(ground_truth, reverse=True)[:valid_k]
    idcg = np.sum(np.array(ideal_ranking) / np.log2(np.arange(2, len(ideal_ranking) + 2)))
    
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(pred_scores, ground_truth, k=10):
    ground_truth = binary_relevance(ground_truth)
    if len(ground_truth) == 0:
        return 0.0
    sorted_indices = np.argsort(-pred_scores)
    ranked_items = ground_truth[sorted_indices[:k]]
    return np.sum(ranked_items) / k

def recall_at_k(pred_scores, ground_truth, k=10):
    ground_truth = binary_relevance(ground_truth)
    if np.sum(ground_truth) == 0:
        return 0.0
    sorted_indices = np.argsort(-pred_scores)
    ranked_items = ground_truth[sorted_indices[:k]]
    return np.sum(ranked_items) / np.sum(ground_truth)

def evaluate_model(model, dataset, k=10):
    """Evaluate metrics for all users."""
    start_time = time.time()
    logger.info("Starting model evaluation...")

    ndcg_scores, precision_scores, recall_scores = [], [], []
    mse_scores, rmse_scores = [], []
    
    device = next(model.parameters()).device
    for task in dataset.create_meta_batch(len(dataset.valid_users)):
        query_x = task['query'][:, 0].long().to(device)
        query_y = task['query'][:, 1].float().to(device)

        with torch.no_grad():
            query_pred = model(query_x)  # Get raw model output

        if query_pred.ndim == 1:  # If model outputs a single score per query
            query_pred = query_pred.unsqueeze(0)  # Convert to 2D tensor

        query_pred = query_pred.cpu().numpy()  # Convert to NumPy array
        true_ratings = query_y.cpu().numpy()
        print(f"Fixed Query Predictions Shape: {query_pred.shape}")
        query_pred = query_pred.squeeze(0)  # Remove batch dimension if it's (1, num_items)

        for i in range(min(len(query_x), len(query_pred))):  # Ensure valid indexing
            pred_scores = query_pred[i]  # Extract score for this item
            ground_truth = true_ratings  # Ensure it's an array with multiple values

            if len(ground_truth) < k:  # Adjust k if needed
                k = len(ground_truth)

            print(f"Sample {i} - Ground Truth: {ground_truth}")
            
            ndcg_scores.append(ndcg_at_k(pred_scores, ground_truth, k))
            precision_scores.append(precision_at_k(pred_scores, ground_truth, k))
            recall_scores.append(recall_at_k(pred_scores, ground_truth, k))
            mse_scores.append(mse(pred_scores, ground_truth))
            rmse_scores.append(rmse(pred_scores, ground_truth))
    
    results = {
        'NDCG@k': np.mean(ndcg_scores),
        'Precision@k': np.mean(precision_scores),
        'Recall@k': np.mean(recall_scores),
        'MSE': np.mean(mse_scores),
        'RMSE': np.mean(rmse_scores),
    }
    
    logger.info(f"Model evaluation completed in {time.time() - start_time:.2f} seconds")
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