import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_preparation import MovieLensMetaDataset
from copy import deepcopy

class MAMLRecommender(nn.Module):
    def __init__(self, n_items, embedding_dim=50, hidden_dim=100):
        super().__init__()
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Prediction network
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, item_ids):
        item_emb = self.item_embeddings(item_ids)
        return self.net(item_emb).squeeze()
    
    def clone_params(self):
        """Clone the model parameters for task-specific adaptation"""
        cloned_params = []
        for param in self.net.parameters():
            cloned_params.append(param.clone())
        return cloned_params
    
    def forward_with_params(self, item_ids, params):
        """Forward pass using the provided parameters"""
        item_emb = self.item_embeddings(item_ids)
        x = item_emb
        
        # Manual forward pass through layers with custom parameters
        start_idx = 0
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                weight = params[start_idx]
                bias = params[start_idx + 1]
                x = F.linear(x, weight, bias)
                start_idx += 2
            if isinstance(layer, nn.ReLU):
                x = F.relu(x)
                
        return x.squeeze()

class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=1):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def inner_loop(self, support_x, support_y):
        """Perform inner loop adaptation"""
        params = self.model.clone_params()
        
        for _ in range(self.num_inner_steps):
            pred = self.model.forward_with_params(support_x, params)
            loss = F.mse_loss(pred, support_y)
            
            # Manual gradient computation and parameter update
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [p - self.inner_lr * g for p, g in zip(params, grads)]
            
        return params
    
    def train_step(self, tasks_batch):
        """Perform one meta-training step"""
        meta_loss = 0
        
        for task in tasks_batch:
            support_x = task['support'][:, 0].long()
            support_y = task['support'][:, 1].float()
            query_x = task['query'][:, 0].long()
            query_y = task['query'][:, 1].float()
            
            # Inner loop adaptation
            adapted_params = self.inner_loop(support_x, support_y)
            
            # Compute loss on query set with adapted parameters
            query_pred = self.model.forward_with_params(query_x, adapted_params)
            meta_loss += F.mse_loss(query_pred, query_y)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() / len(tasks_batch)

def train_maml(dataset, n_epochs=50, tasks_per_batch=4, eval_interval=5):
    """Main training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and MAML trainer
    model = MAMLRecommender(n_items=dataset.n_items).to(device)
    maml = MAML(model)
    
    print(f"Starting training on {device}")
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 20  # Number of meta-batches per epoch
        
        for _ in range(n_batches):
            # Sample batch of tasks
            tasks_batch = dataset.create_meta_batch(tasks_per_batch)
            
            # Move tasks to device
            for task in tasks_batch:
                task['support'] = task['support'].to(device)
                task['query'] = task['query'].to(device)
            
            # Perform meta-update
            batch_loss = maml.train_step(tasks_batch)
            epoch_loss += batch_loss
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % eval_interval == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    return model

# Example usage
def main():
    # Initialize dataset
    dataset = MovieLensMetaDataset(
        data_path='data/ml-32m/ratings.csv',
        n_support=10,
        n_query=5
    )
    
    # Train model
    model = train_maml(
        dataset=dataset,
        n_epochs=50,
        tasks_per_batch=4
    )
    
    # Save trained model
    torch.save(model.state_dict(), 'maml_recommender.pth')
    
    # Example prediction for a user
    test_user = dataset.valid_users[0]
    test_task = dataset.create_user_task(test_user)
    
    device = next(model.parameters()).device
    support_x = test_task['support'][:, 0].long().to(device)
    support_y = test_task['support'][:, 1].float().to(device)
    query_x = test_task['query'][:, 0].long().to(device)
    
    # Create MAML instance for prediction
    maml = MAML(model)
    
    # Adapt to test user
    adapted_params = maml.inner_loop(support_x, support_y)
    
    # Get predictions
    with torch.no_grad():
        predictions = model.forward_with_params(query_x, adapted_params)
        print("\nTest User Predictions:")
        print(predictions.cpu().numpy())

if __name__ == "__main__":
    main()
# Refactor: 2024-10-31 - 2/3

# Optimize: 2024-11-01 - 3/3
