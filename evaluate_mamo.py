import torch
from data_preparation import MovieLensMetaDataset
from evaluation import evaluate_model
from mamo import MAMORecommender

if __name__ == "__main__":
    # Initialize dataset
    dataset = MovieLensMetaDataset(
        data_path='data/ml-32m/ratings.csv',
        n_support=10,
        n_query=5
    )
    
    # Load trained MAMO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MAMORecommender(
        n_items=dataset.n_items,
        hidden_dim=200,  # Match the hidden dimension in the checkpoint
        memory_size=200  # Match the memory size in the checkpoint
    ).to(device)  # Move model to the correct device
    model.load_state_dict(torch.load('mamo_recommender.pth', weights_only=True))
    
    # Evaluate the model
    results = evaluate_model(model, dataset, k=10)
    
    # Print results
    print("MAMO Model Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")