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
labels = label_encoder.fit_transform(superfamilies)

# Filter labels with at least 2 samples
print("Filtering labels with at least 2 samples...")
unique, counts = np.unique(labels, return_counts=True)
valid_labels = unique[counts > 1]
mask = np.isin(labels, valid_labels)
embeddings = embeddings[mask]
labels = labels[mask]
print(f"Filtered embeddings shape: {embeddings.shape}")

# Split data
print("Splitting data into train and test sets...")
train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
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

# Triplet Sampling Function
def get_triplets(embeddings, labels, num_triplets=10000):
    triplets = []
    label_dict = {}
    
    for i, label in enumerate(labels):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(i)
    
    for _ in range(num_triplets):
        anchor_idx = np.random.randint(0, len(labels))
        anchor_label = labels[anchor_idx]
        
        positive_idx = np.random.choice(label_dict[anchor_label])
        
        negative_label = np.random.choice([l for l in label_dict.keys() if l != anchor_label])
        negative_idx = np.random.choice(label_dict[negative_label])
        
        triplets.append((anchor_idx, positive_idx, negative_idx))
    
    return triplets

# Projection Head Model
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=256):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.projection(x)

# Training Loop
def train_projection_head(train_embeddings, train_labels, test_embeddings, test_labels, epochs=10, batch_size=256, lr=0.001, num_triplets=200000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=0.3)

    train_dataset = ProteinDataset(train_embeddings, train_labels)
    test_dataset = ProteinDataset(test_embeddings, test_labels)




    #
    # BRUH BRUH BRUH
    #
    # why tf would i use the same triplets over and over again?
    #
    #   idiot
    #
    #

    print("Generating triplets...")
    train_triplets = get_triplets(train_embeddings, train_labels, num_triplets=num_triplets)
    test_triplets = get_triplets(test_embeddings, test_labels, num_triplets=int(num_triplets*0.05))
    print(f"Triplets generated: {len(train_triplets)} train, {len(test_triplets)} test")

    # Initialize best model tracking
    best_test_loss = float('inf')
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        total_loss = 0
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

        # Evaluation on test set
        model.eval()
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

        # Save best model if test loss improves
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model found! Test loss: {best_test_loss:.4f}")

    # Load the best model state before returning
    model.load_state_dict(best_model_state)
    return model

# Main execution
if __name__ == "__main__":
    print("Starting training process...")
    
    # Train the model
    projection_model = train_projection_head(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        epochs=25, batch_size=256, lr=0.001, 
        num_triplets=200000
    )

    # Save best model
    output_dir = "/scratch/gpfs/jr8867/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "projection_model_200k_best.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")
