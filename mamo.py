import torch
import torch.nn as nn
import torch.nn.functional as F
from data_preparation import MovieLensMetaDataset
from maml import MAMLRecommender, MAML

class MemoryModule(nn.Module):
    def __init__(self, memory_size, embedding_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
    
    def forward(self, task_embedding):
        # Compute attention scores
        scores = torch.matmul(task_embedding.unsqueeze(0), self.memory.transpose(0, 1))
        attention = F.softmax(scores, dim=-1)
        
        # Retrieve memory
        retrieved_memory = torch.matmul(attention, self.memory)
        return retrieved_memory

class MAMORecommender(nn.Module):
    """
    Ensemble model combining MAML and MAMO.
    """
    def __init__(self, n_items, embedding_dim=50, hidden_dim=100, memory_size=100):
        super().__init__()
        self.maml_recommender = MAMLRecommender(n_items, embedding_dim, hidden_dim)
        self.memory_module = MemoryModule(memory_size, embedding_dim)
        
        # Combination network with dropout
        self.combination_network = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout with 50% probability
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, item_ids, task_embedding=None):
        """
        Forward pass for the ensemble model.
        """
        # Get MAML predictions
        maml_output = self.maml_recommender(item_ids)  # Shape: [batch_size]
        
        # Retrieve memory if task_embedding is provided
        if task_embedding is not None:
            memory_output = self.memory_module(task_embedding)  # Shape: [1, embedding_dim]
            
            # Remove the extra dimension and repeat to match the batch size
            memory_output = memory_output.squeeze(0).repeat(item_ids.size(0), 1)  # Shape: [batch_size, embedding_dim]
            
            # Combine MAML output and memory output
            combined_input = torch.cat([maml_output.unsqueeze(1), memory_output], dim=-1)  # Shape: [batch_size, embedding_dim + 1]
            
            # Pass through the combination network
            combined_output = self.combination_network(combined_input)  # Shape: [batch_size, 1]
            return combined_output.squeeze()  # Shape: [batch_size]
        else:
            return maml_output
        
class MAMO:
    """
    MAMO trainer combining MAML and memory-augmented meta-optimization.
    """
    def __init__(self, model, inner_lr=0.0005, meta_lr=0.000005, num_inner_steps=1):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr, weight_decay=1e-5)

    def inner_loop(self, support_x, support_y):
        """
        Perform inner loop adaptation with memory augmentation.
        """
        params = self.model.maml_recommender.clone_params()
        
        for _ in range(self.num_inner_steps):
            pred = self.model.maml_recommender.forward_with_params(support_x, params)
            loss = F.mse_loss(pred, support_y)
            
            # Manual gradient computation and parameter update
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [p - self.inner_lr * g for p, g in zip(params, grads)]
        
        return params

    def train_step(self, tasks_batch):
        """
        Perform one meta-training step with memory augmentation.
        """
        meta_loss = 0
        
        for task in tasks_batch:
            support_x = task['support'][:, 0].long()
            support_y = task['support'][:, 1].float()
            query_x = task['query'][:, 0].long()
            query_y = task['query'][:, 1].float()
            
            # Inner loop adaptation
            adapted_params = self.inner_loop(support_x, support_y)
            
            # Compute task embedding (e.g., average of support set embeddings)
            task_embedding = self.model.maml_recommender.item_embeddings(support_x).mean(dim=0)
            
            # Compute loss on query set with adapted parameters and memory
            query_pred = self.model(query_x, task_embedding)
            meta_loss += F.mse_loss(query_pred, query_y)
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() / len(tasks_batch)

def train_mamo(dataset, n_epochs=100, tasks_per_batch=4, eval_interval=5):
    """
    Main training loop for MAMO.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and MAMO trainer
    model = MAMORecommender(
        n_items=dataset.n_items,
        hidden_dim=200,  # Increase hidden dimension
        memory_size=200  # Increase memory size
    ).to(device)
    mamo = MAMO(model, inner_lr=0.000001, meta_lr=0.00000001, num_inner_steps=5)
    
    # Training loop
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
            batch_loss = mamo.train_step(tasks_batch)
            epoch_loss += batch_loss
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % eval_interval == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save trained model
    torch.save(model.state_dict(), 'mamo_recommender.pth')
    return model



# Example usage
if __name__ == "__main__":
    # Initialize dataset
    dataset = MovieLensMetaDataset(
        data_path='data/ml-32m/ratings.csv',
        n_support=10,
        n_query=5
    )
    
    # Train model
    model = train_mamo(
        dataset=dataset,
        n_epochs=50,
        tasks_per_batch=4
    )
    
    # Save trained model
    torch.save(model.state_dict(), 'mamo_recommender.pth')