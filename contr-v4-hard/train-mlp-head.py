import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import os
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Func: Fetching Data from GPFS
def fetch_data(arr_type):
    if arr_type != 'train' and arr_type != 'test':
        raise ValueError("arr_type must be 'train' or 'test'")
    
    embeddings = np.load(f'/scratch/gpfs/jr8867/main/db/train-test-fold/{arr_type}_embeddings.npy')
    indicies = np.load(f'/scratch/gpfs/jr8867/main/db/train-test-fold/{arr_type}_indicies.npy')
    superfamilies = np.load(f'/scratch/gpfs/jr8867/main/db/train-test-fold/{arr_type}_superfamilies.npy')
    families = np.load(f'/scratch/gpfs/jr8867/main/db/train-test-fold/{arr_type}_families.npy')
    folds = np.load(f'/scratch/gpfs/jr8867/main/db/train-test-fold/{arr_type}_folds.npy')
    return embeddings, indicies, superfamilies, families, folds

# Load Data

print("Fetching data...")

train_embeddings, _, train_superfamilies, train_families, train_folds = fetch_data('train')
test_embeddings, _, test_superfamilies, test_families, test_folds = fetch_data('test')

train_embeddings_faiss = faiss.read_index('/scratch/gpfs/jr8867/main/db/train-test-fold/train_embeddings.index')
test_embeddings_faiss = faiss.read_index('/scratch/gpfs/jr8867/main/db/train-test-fold/test_embeddings.index')

print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Train superfamilies shape: {train_superfamilies.shape}")
print(f"Train families shape: {train_families.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")
print(f"Test superfamilies shape: {test_superfamilies.shape}")
print(f"Test families shape: {test_families.shape}")
print(f"Train folds shape: {train_folds.shape}")
print(f"Test folds shape: {test_folds.shape}")

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, embeddings, superfamilies, families, folds):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.superfamilies = torch.tensor(superfamilies, dtype=torch.long)
        self.families = torch.tensor(families, dtype=torch.long)
        self.folds = torch.tensor(folds, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.superfamilies[idx], self.families[idx], self.folds[idx]

# Triplet Sampling Function
def get_triplets(embeddings, embeddings_faiss, superfamilies, families, folds, num_triplets=10000, k=1000, hard_negative_ratio=0.05):
    """
    Sample triplets using hard negative mining with FAISS.
    
    Args:
        embeddings: Protein embeddings
        embeddings_faiss: FAISS index of embeddings
        superfamilies: Array of superfamily labels
        families: Array of family labels
        folds: Array of fold labels
        num_triplets: Total number of triplets to generate
        k: Number of nearest neighbors to retrieve for hard negative mining
        hard_negative_ratio: Proportion of hard negatives vs random negatives
        
    Returns:
        Array of triplets [anchor_idx, positive_idx, negative_idx]
    """
    triplets = np.zeros((num_triplets, 3), dtype=np.int64)
    label_dict = {}
    
    # Build dictionary mapping labels to indices
    for i, label in enumerate(superfamilies):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(i)
    
    # Calculate number of hard negatives vs random negatives
    num_hard_triplets = int(num_triplets * hard_negative_ratio)
    num_random_triplets = num_triplets - num_hard_triplets
    
    # Generate hard negative triplets
    hard_triplet_idx = 0
    with tqdm(total=num_hard_triplets, desc="Getting hard triplets", unit="triplet") as pbar:
        while hard_triplet_idx < num_hard_triplets:
            # Select random anchor
            anchor_idx = np.random.randint(0, len(superfamilies))
            anchor_label = superfamilies[anchor_idx]
            anchor_embedding = embeddings[anchor_idx].reshape(1, -1).astype(np.float32)
            
            # Select positive from same superfamily
            positive_candidates = [idx for idx in label_dict[anchor_label] if idx != anchor_idx]
            if not positive_candidates:
                continue  # Skip if no positive candidates
            positive_idx = np.random.choice(positive_candidates)
            
            # Find hard negatives using FAISS
            _, nn_indices = embeddings_faiss.search(anchor_embedding, k)
            nn_indices = nn_indices[0]  # Flatten
            
            # Find indices where superfamily differs (hard negatives)
            hard_negative_candidates = []
            for nn_idx in nn_indices:
                if superfamilies[nn_idx] != anchor_label:
                    hard_negative_candidates.append(nn_idx)
            
            if not hard_negative_candidates:
                continue  # Skip if no hard negative candidates
            
            # Choose a hard negative
            negative_idx = np.random.choice(hard_negative_candidates)
            
            triplets[hard_triplet_idx] = [anchor_idx, positive_idx, negative_idx]
            hard_triplet_idx += 1
            pbar.update(1)
    
    # Generate random triplets for the remainder
    with tqdm(total=num_random_triplets, desc="Getting random triplets", unit="triplet") as pbar:
        for i in range(num_hard_triplets, num_triplets):
            anchor_idx = np.random.randint(0, len(superfamilies))
            anchor_label = superfamilies[anchor_idx]
            
            # Select positive from same superfamily
            positive_candidates = [idx for idx in label_dict[anchor_label] if idx != anchor_idx]
            if not positive_candidates:
                continue  # Skip if no positive candidates
            positive_idx = np.random.choice(positive_candidates)
            
            # Select negative from different superfamily
            negative_label = np.random.choice([l for l in label_dict.keys() if l != anchor_label])
            negative_idx = np.random.choice(label_dict[negative_label])
            
            triplets[i] = [anchor_idx, positive_idx, negative_idx]
            pbar.update(1)
    
    np.random.shuffle(triplets)
    return triplets


# Projection Head Model
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(ProjectionHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)

# Training Loop
def train_projection_head(train_embeddings, train_embeddings_faiss, train_superfamilies, train_families, train_folds, 
                         test_embeddings, test_embeddings_faiss, test_superfamilies, test_families, test_folds, 
                         output_dir, model_name, epochs=10, lr=0.001, triplet_margin=0.2, 
                         train_triplet_count=200000, test_triplet_count=10000, batch_size=32,
                         initial_model_path=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=256).to(device)
    
    # Load initial model if provided
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading initial model from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=triplet_margin)

    train_dataset = ProteinDataset(train_embeddings, train_superfamilies, train_families, train_folds)
    test_dataset = ProteinDataset(test_embeddings, test_superfamilies, test_families, test_folds)

    best_loss = float('inf')

    for epoch in range(epochs):

        # Get triplets
        train_triplets = get_triplets(
            train_embeddings, train_embeddings_faiss, train_superfamilies, 
            train_families, train_folds, num_triplets=train_triplet_count
        )
        
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(train_triplets), batch_size), desc=f"Epoch {epoch+1}", unit="batch"):
            batch_triplets = train_triplets[i:i + batch_size]
            anchors = torch.stack([train_dataset[anchor_idx][0] for anchor_idx, _, _ in batch_triplets]).to(device)
            positives = torch.stack([train_dataset[pos_idx][0] for _, pos_idx, _ in batch_triplets]).to(device)
            negatives = torch.stack([train_dataset[neg_idx][0] for _, _, neg_idx in batch_triplets]).to(device)

            anchor_out = model(anchors)
            positive_out = model(positives)
            negative_out = model(negatives)

            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_triplets)

        # Save model if it has the best loss so far
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model_path = os.path.join(output_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with loss: {best_loss:.4e}")

        # Evaluation on test set
        test_triplets = get_triplets(
            test_embeddings, test_embeddings_faiss, test_superfamilies, 
            test_families, test_folds, num_triplets=test_triplet_count
        )

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for anchor_idx, pos_idx, neg_idx in test_triplets:
                anchor = test_dataset[anchor_idx][0].to(device)
                positive = test_dataset[pos_idx][0].to(device)
                negative = test_dataset[neg_idx][0].to(device)

                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                t_loss = criterion(anchor_out, positive_out, negative_out)
                total_test_loss += t_loss.item()
        avg_test_loss = total_test_loss / len(test_triplets)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4e} | Test Loss: {avg_test_loss:.4e}")

    return model, best_loss

# Main execution
if __name__ == "__main__":
    print("Starting training process...")
    
    output_dir = "/scratch/gpfs/jr8867/main/contr-v4-hard/models"

    # Start time
    start_time = time.time()
    
    # Train the model from scratch
    projection_model, best_loss = train_projection_head(
        train_embeddings, train_embeddings_faiss, train_superfamilies, train_families, train_folds,
        test_embeddings, test_embeddings_faiss, test_superfamilies, test_families, test_folds,
        output_dir=output_dir, model_name="contr-v4-large-hard",
        epochs=100, lr=0.001, triplet_margin=0.2, 
        train_triplet_count=200000, test_triplet_count=10000, 
        batch_size=8192
    )

    end_time = time.time()
    print(f"Epoch completed in {end_time - start_time:.2f} seconds")

    # Save the model
    model_path = os.path.join(output_dir, "contr-v4-large-hard-end.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print(f"Training completed. Best loss achieved: {best_loss:.4e}")
