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
        # Optionally normalize final embeddings
        if self.normalize_output:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

# Training Loop
def train_projection_head(train_embeddings, train_labels, test_embeddings, test_labels, epochs=10, batch_size=256, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=128, normalize_output=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=0.2)

    train_dataset = ProteinDataset(train_embeddings, train_labels)
    test_dataset = ProteinDataset(test_embeddings, test_labels)

    print("Generating triplets...")
    train_triplets = get_triplets(train_embeddings, train_labels, num_triplets=200000)
    test_triplets = get_triplets(test_embeddings, test_labels, num_triplets=10000)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        total_loss = 0
        for anchor_idx, pos_idx, neg_idx in train_triplets:
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

    return model

# Main execution
if __name__ == "__main__":
    print("Starting training process...")
    
    # Train the model
    projection_model = train_projection_head(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        epochs=25, batch_size=256, lr=0.001
    )

    # Save model
    output_dir = "/scratch/gpfs/jr8867/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "projection_model.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
