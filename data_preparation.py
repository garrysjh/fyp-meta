import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

class MovieLensMetaDataset:
    def __init__(self, data_path, n_support=10, n_query=5):
        """
        Initialize MovieLens data set for Meta Learning
        """
        self.n_support = n_support
        self.n_query = n_query
        
        # Load and preprocess data
        self.ratings_df = pd.read_csv(data_path, sep=',', engine='python')
        
        # Normalize ratings to [0, 1]
        self.ratings_df['rating'] = self.ratings_df['rating'] / 5.0
        
        # Create user and item mappings
        self.user_id_map = {id: idx for idx, id in enumerate(self.ratings_df['userId'].unique())}
        self.movie_id_map = {id: idx for idx, id in enumerate(self.ratings_df['movieId'].unique())}

        # Create n_items attribute (unique items[movies]) in dataset
        self.n_items = len(self.movie_id_map)
        
        # Map IDs to indices
        self.ratings_df['userId'] = self.ratings_df['userId'].map(self.user_id_map)
        self.ratings_df['movieId'] = self.ratings_df['movieId'].map(self.movie_id_map)
        
        # Create user-item dictionary
        self.user_items = defaultdict(list)
        for row in self.ratings_df.itertuples():
            self.user_items[row.userId].append({
                'movie': row.movieId,
                'rating': row.rating,
                'timestamp': row.timestamp
            })
            
        # Filter users with sufficient interactions
        min_items = n_support + n_query
        self.valid_users = [user for user, items in self.user_items.items() 
                           if len(items) >= min_items]
        
    def create_user_task(self, user_id):
        """
        Create a task (support and query sets) for a specific user
        """
        user_data = self.user_items[user_id]
        
        # Sort by timestamp for temporal split
        user_data.sort(key=lambda x: x['timestamp'])
        
        # Select support and query items
        support_items = user_data[:self.n_support]
        query_items = user_data[-self.n_query:]
        
        # Create tensors
        support_set = torch.tensor([(item['movie'], item['rating']) 
                                  for item in support_items])
        query_set = torch.tensor([(item['movie'], item['rating']) 
                                for item in query_items])
        
        return {
            'support': support_set,
            'query': query_set,
            'user_id': user_id
        }
    
    def create_meta_batch(self, batch_size):
        """
        Create a batch of tasks for meta-learning
        """
        selected_users = np.random.choice(self.valid_users, 
                                        size=batch_size, 
                                        replace=False)
        return [self.create_user_task(user) for user in selected_users]

def prepare_meta_batch(batch, device):
    """
    Prepare a meta-batch for training
    """
    support_x = torch.stack([task['support'][:, 0] for task in batch]).to(device)
    support_y = torch.stack([task['support'][:, 1] for task in batch]).to(device)
    query_x = torch.stack([task['query'][:, 0] for task in batch]).to(device)
    query_y = torch.stack([task['query'][:, 1] for task in batch]).to(device)
    
    return {
        'support_x': support_x,
        'support_y': support_y,
        'query_x': query_x,
        'query_y': query_y
    }

# Example usage:
if __name__ == "__main__":
    # Initialize dataset
    dataset = MovieLensMetaDataset(
        data_path='data/ml-32m/ratings.csv',
        n_support=10,
        n_query=5
    )

    # Verify it loaded correctly
    print(f"Number of ratings loaded: {len(dataset.ratings_df)}")
    print(f"Number of users: {len(dataset.user_id_map)}")
    print(f"Number of movies: {len(dataset.movie_id_map)}")
        
    # Create a meta-batch
    meta_batch = dataset.create_meta_batch(batch_size=4)
    
    # Prepare for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_data = prepare_meta_batch(meta_batch, device)
    
    print(f"Support set shape: {batch_data['support_x'].shape}")
    print(f"Query set shape: {batch_data['query_x'].shape}")