#!/usr/bin/env python

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

# Fetching Data from Storages
def fetch_full_arr(arr_type):
    return np.load(f'/scratch/gpfs/jr8867/embeddings/scop/{arr_type}.npy')

# Load data
print("Loading embeddings and indices...")
embeddings = fetch_full_arr('embeddings')
indices = fetch_full_arr('indices')
print(f"Embeddings shape: {embeddings.shape}")
print(f"Indices shape: {indices.shape}")

# Load metadata
print("Loading metadata...")
metadata_df = pd.read_csv('/scratch/gpfs/jr8867/datasets/scop/scop_data.csv')

# Get superfamily information
print("Extracting superfamily information...")
superfamilies = np.array([metadata_df.loc[metadata_df['index'] == i, 'sf'].values[0] for i in indices])

# Encode labels
label_encoder = LabelEncoder()
full_labels = label_encoder.fit_transform(superfamilies)  # Keep full labels

# Filter labels with at least 2 samples
print("Filtering labels with at least 2 samples...")
unique, counts = np.unique(full_labels, return_counts=True)
valid_labels = unique[counts > 1]
mask = np.isin(full_labels, valid_labels)

# Store the mapping from filtered indices to original indices
filtered_to_full_idx = np.where(mask)[0]  # This maps our filtered indices back to full dataset indices
embeddings = embeddings[mask]
labels = full_labels[mask]
print(f"Filtered embeddings shape: {embeddings.shape}")

# Split data
print("Splitting data into train and test sets...")
train_embeddings, test_embeddings, train_labels, test_labels, train_idx_map, test_idx_map = train_test_split(
    embeddings, labels, filtered_to_full_idx, test_size=0.2, random_state=42, stratify=labels
)

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Projection Head Model
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128, normalize_output=True):
        super(ProjectionHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
        self.normalize_output = normalize_output
    
    def forward(self, x):
        x = self.model(x)
        if self.normalize_output:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

def get_transformed_embeddings(model, embeddings, batch_size=256, device='cuda'):
    """Transform embeddings through the pretrained model."""
    model.eval()
    transformed = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(device)
            out = model(batch).cpu().numpy()
            transformed.append(out)
    
    return np.vstack(transformed)

def get_triplets_faiss(embeddings, transformed_embeddings, labels, idx_map, full_labels, refined_index, k=25, num_triplets=10000):
    """
    Generate triplets using FAISS-based negative mining in the transformed space.
    Uses transformed embeddings for neighbor search but returns indices for original embeddings.
    
    Parameters:
    -----------
    idx_map : np.ndarray
        Maps indices in our current subset back to indices in the full dataset
    full_labels : np.ndarray
        Labels for the full dataset (needed for FAISS results)
    """
    triplets = []
    label_dict = {}
    for i, lbl in enumerate(labels):
        if lbl not in label_dict:
            label_dict[lbl] = []
        label_dict[lbl].append(i)

    N = len(labels)
    subset_to_full = {i: full_idx for i, full_idx in enumerate(idx_map)}

    for _ in tqdm(range(num_triplets), desc=f"Generating triplets"):
        # Random anchor
        anchor_idx = np.random.randint(0, N)
        anchor_label = labels[anchor_idx]
        
        # Use transformed embedding for search
        anchor_emb = transformed_embeddings[anchor_idx]

        # Positive from same label
        positive_idx = np.random.choice(label_dict[anchor_label])

        # Query FAISS for top-k neighbors using transformed embeddings
        _, I = refined_index.search(anchor_emb.reshape(1, -1), k)
        neighbors = I[0]

        # Find mislabeled neighbors
        # Convert FAISS results (which reference full dataset) to our subset indices
        valid_neighbors = []
        for n in neighbors:
            if n < len(full_labels) and full_labels[n] != anchor_label:  # Check against full labels
                # Try to find this index in our mapping
                for subset_idx, full_idx in subset_to_full.items():
                    if full_idx == n:
                        valid_neighbors.append(subset_idx)
                        break

        # Add triplets for valid mislabeled neighbors
        for neg_idx in valid_neighbors:
            triplets.append((anchor_idx, positive_idx, neg_idx))

        # Add random negatives if needed
        needed = k - len(valid_neighbors)
        if needed > 0:
            negative_candidates = np.where(labels != anchor_label)[0]
            if len(negative_candidates) > 0:  # Make sure we have candidates
                random_negatives = np.random.choice(negative_candidates, size=min(needed, len(negative_candidates)), replace=False)
                for neg_idx in random_negatives:
                    triplets.append((anchor_idx, positive_idx, neg_idx))

    return triplets

def fine_tune_projection_head(train_embeddings, train_labels, test_embeddings, test_labels, 
                            train_idx_map, test_idx_map, full_labels,
                            pretrained_path, refined_index_path, epochs=10, batch_size=256, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and load pretrained weights
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=128, normalize_output=True).to(device)
    print(f"Loading pretrained model from {pretrained_path}")
    model.load_state_dict(torch.load(pretrained_path))
    
    # Load the refined FAISS index for negative mining
    print(f"Loading refined FAISS index from {refined_index_path}")
    refined_index = faiss.read_index(refined_index_path)
    
    # Transform embeddings through pretrained model for FAISS search
    print("Transforming embeddings through pretrained model...")
    train_transformed = get_transformed_embeddings(model, train_embeddings, device=device)
    test_transformed = get_transformed_embeddings(model, test_embeddings, device=device)
    
    # Use a lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=0.2)

    # Create datasets
    train_dataset = ProteinDataset(train_embeddings, train_labels)
    test_dataset = ProteinDataset(test_embeddings, test_labels)

    best_test_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Generate triplets using FAISS on transformed embeddings
        print(f"Epoch {epoch+1}: Generating training triplets...")
        train_triplets = get_triplets_faiss(
            train_embeddings, train_transformed, train_labels, 
            train_idx_map, full_labels, refined_index,
            k=10, num_triplets=10000
        )

        # Training loop
        for anchor_idx, pos_idx, neg_idx in tqdm(train_triplets, desc=f"Epoch {epoch+1} [Train]"):
            anchor = train_dataset[anchor_idx][0].to(device)
            positive = train_dataset[pos_idx][0].to(device)
            negative = train_dataset[neg_idx][0].to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_triplets)

        # Evaluation
        model.eval()
        print(f"Generating test triplets...")
        test_triplets = get_triplets_faiss(
            test_embeddings, test_transformed, test_labels,
            test_idx_map, full_labels, refined_index,
            k=10, num_triplets=2000
        )
        
        total_test_loss = 0.0
        with torch.no_grad():
            for anchor_idx, pos_idx, neg_idx in tqdm(test_triplets, desc=f"Epoch {epoch+1} [Test]"):
                anchor = test_dataset[anchor_idx][0].to(device)
                positive = test_dataset[pos_idx][0].to(device)
                negative = test_dataset[neg_idx][0].to(device)

                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                t_loss = criterion(anchor_out, positive_out, negative_out)
                total_test_loss += t_loss.item()

        avg_test_loss = total_test_loss / len(test_triplets)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            print(f"New best model found! Test Loss: {best_test_loss:.4f}")

    # Return the best model
    model.load_state_dict(best_model_state)
    return model

if __name__ == "__main__":
    print("Starting fine-tuning process...")
    
    # Paths to models and indices
    pretrained_path = "/scratch/gpfs/jr8867/models/projection_model_old.pth"
    refined_index_path = '/scratch/gpfs/jr8867/embeddings/scop/protein_embeddings_refined_old.index'
    
    # Fine-tune the model
    projection_model = fine_tune_projection_head(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        train_idx_map, test_idx_map, full_labels,
        pretrained_path=pretrained_path,
        refined_index_path=refined_index_path,
        epochs=15,
        batch_size=256,
        lr=0.0001  # Lower learning rate for fine-tuning
    )

    # Save fine-tuned model
    output_dir = "/scratch/gpfs/jr8867/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "projection_model_finetuned.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Fine-tuned model saved to {model_path}")
