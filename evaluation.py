import torch
import numpy as np
from data_preparation import MovieLensMetaDataset
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set logging level for detailed information

def ndcg_at_k(ranked_items, ground_truth, k=10):
    """Calculate NDCG@k for a single user's ranked list of items."""
    dcg, idcg = 0.0, 0.0
    for i in range(k):
        if i < len(ranked_items):
            if ranked_items[i] in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # 1/log2(i+2)
        if i < len(ground_truth):
            idcg += 1.0 / np.log2(i + 2)  # 1/log2(i+2)
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_items, ground_truth, k=10):
    """Calculate Precision@k for a single user's ranked list of items."""
    hits = sum([1 for i in ranked_items[:k] if i in ground_truth])
    return hits / k

def recall_at_k(ranked_items, ground_truth, k=10):
    """Calculate Recall@k for a single user's ranked list of items."""
    hits = sum([1 for i in ranked_items[:k] if i in ground_truth])
    return hits / len(ground_truth) if len(ground_truth) > 0 else 0.0

def evaluate_model(model, dataset, k=10):
    """Evaluate NDCG@k, Precision@k, and Recall@k for all users in the test set."""
    start_time = time.time()  # Start timer before iterating through users
    logger.info("Starting model evaluation...")

    ndcg_scores, precision_scores, recall_scores = [], [], []
    
    for user in dataset.valid_users:
        # Create the test task for the user
        start_user_time = time.time()

        task = dataset.create_user_task(user)
        
        # Get all items and predict scores
        item_ids = torch.arange(dataset.n_items).long()
        item_scores = model(item_ids).detach().cpu()
        
        # Rank items based on predicted scores
        _, ranked_items = item_scores.sort(descending=True)
        
        # Use actual items in query set as ground truth
        query_set = task['query'][:, 0].cpu().numpy()  # items in query set
        
        # Calculate metrics
        ndcg_score = ndcg_at_k(ranked_items[:k].numpy(), query_set, k)
        precision_score = precision_at_k(ranked_items[:k].numpy(), query_set, k)
        recall_score = recall_at_k(ranked_items[:k].numpy(), query_set, k)
        
        ndcg_scores.append(ndcg_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)

        # Log user results
        end_user_time = time.time()
        logger.debug(f"Processed user {user} in {end_user_time - start_user_time:.2f} seconds")
    
    end_time = time.time()
    logger.info(f"Model evaluation completed in {end_time - start_time:.2f} seconds")
    # Average scores across users
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    
    return {
        'NDCG@k': avg_ndcg,
        'Precision@k': avg_precision,
        'Recall@k': avg_recall
    }

# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = MovieLensMetaDataset(
        data_path='data/ml-32m/ratings.csv',
        n_support=10,
        n_query=5
    )
    
    # Load trained model
    from maml import MAMLRecommender  # Import your MAML model class
    model = MAMLRecommender(n_items=dataset.n_items)
    
    # Load the model's state dictionary safely
    try:
        model.load_state_dict(torch.load('maml_recommender.pth', weights_only=True))
    except TypeError:
        # For older PyTorch versions where weights_only is unsupported
        model.load_state_dict(torch.load('maml_recommender.pth'))
    
    # Evaluate the model
    results = evaluate_model(model, dataset, k=10)
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")