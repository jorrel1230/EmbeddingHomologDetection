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
    
    embeddings = np.load(f'/scratch/gpfs/jr8867/main/db/train-test/{arr_type}_embeddings.npy')
    indicies = np.load(f'/scratch/gpfs/jr8867/main/db/train-test/{arr_type}_indicies.npy')
    superfamilies = np.load(f'/scratch/gpfs/jr8867/main/db/train-test/{arr_type}_superfamilies.npy')
    families = np.load(f'/scratch/gpfs/jr8867/main/db/train-test/{arr_type}_families.npy')
    folds = np.load(f'/scratch/gpfs/jr8867/main/db/train-test-fold/{arr_type}_folds.npy')
    return embeddings, indicies, superfamilies, families, folds

# Load Data

print("Fetching data...")

train_embeddings, _, train_superfamilies, train_families, train_folds = fetch_data('train')
test_embeddings, _, test_superfamilies, test_families, test_folds = fetch_data('test')

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
def get_triplets(embeddings, superfamilies, families, folds, num_triplets=10000):
    triplets = np.zeros((num_triplets, 3), dtype=np.int64)
    label_dict = {}
    
    for i, label in enumerate(superfamilies):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(i)
    
    for i in tqdm(range(num_triplets), desc="Getting triplets", unit="triplet"):
        anchor_idx = np.random.randint(0, len(superfamilies))
        anchor_label = superfamilies[anchor_idx]
        
        positive_idx = np.random.choice(label_dict[anchor_label])
        
        negative_label = np.random.choice([l for l in label_dict.keys() if l != anchor_label])
        negative_idx = np.random.choice(label_dict[negative_label])
        
        triplets[i] = [anchor_idx, positive_idx, negative_idx]
    
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
def train_projection_head(train_embeddings, train_superfamilies, train_families, train_folds, 
                         test_embeddings, test_superfamilies, test_families, test_folds, 
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
        train_triplets = get_triplets(train_embeddings, train_superfamilies, train_families, train_folds, num_triplets=train_triplet_count)
        
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
        test_triplets = get_triplets(test_embeddings, test_superfamilies, test_families, test_folds, num_triplets=test_triplet_count)

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
    
    output_dir = "/scratch/gpfs/jr8867/main/contr-v2/models"
    
    # Train the model from scratch
    projection_model, best_loss = train_projection_head(
        train_embeddings, train_superfamilies, train_families, train_folds,
        test_embeddings, test_superfamilies, test_families, test_folds,
        output_dir=output_dir, model_name="contr-v2-large",
        epochs=30, lr=0.001, triplet_margin=0.2, 
        train_triplet_count=2000000, test_triplet_count=100000, 
        batch_size=8192
    )

    # Save the model
    model_path = os.path.join(output_dir, "contr-v2-large-end.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print(f"Training completed. Best loss achieved: {best_loss:.4e}")
