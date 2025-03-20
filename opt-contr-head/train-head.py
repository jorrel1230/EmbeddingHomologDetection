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
def fetch_train_test_split():
    data_dir = '/scratch/gpfs/jr8867/embeddings/scop/train-test-split'

    train_embeddings = np.load(f'{data_dir}/train_embeddings.npy')
    test_embeddings = np.load(f'{data_dir}/test_embeddings.npy')
    train_labels = np.load(f'{data_dir}/train_labels.npy')
    test_labels = np.load(f'{data_dir}/test_labels.npy')

    return train_embeddings, test_embeddings, train_labels, test_labels

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# # Triplet Sampling Function
# def get_triplets(embeddings, labels, num_triplets=10000):
#     triplets = []
#     label_dict = {}
    
#     for i, label in enumerate(labels):
#         if label not in label_dict:
#             label_dict[label] = []
#         label_dict[label].append(i)
    
#     for _ in range(num_triplets):
#         anchor_idx = np.random.randint(0, len(labels))
#         anchor_label = labels[anchor_idx]
        
#         positive_idx = np.random.choice(label_dict[anchor_label])
        
#         negative_label = np.random.choice([l for l in label_dict.keys() if l != anchor_label])
#         negative_idx = np.random.choice(label_dict[negative_label])
        
#         triplets.append((anchor_idx, positive_idx, negative_idx))
    
#     return triplets


import faiss
import numpy as np

def get_triplets_faiss(embeddings, labels, index, k=25, num_triplets=10000):
    """
    Generate triplets (anchor_idx, positive_idx, negative_idx) using a FAISS index
    for negative mining.

    1) Randomly select an anchor.
    2) Randomly select a positive from the same label.
    3) FAISS-query the anchor for top-k neighbors.
       - If some neighbors have a different label (mislabel), we add each such neighbor as a negative.
       - We then add enough random negatives to bring the total negative count to k.
         If no neighbors are mislabelled, we add k random negatives.

    embeddings : np.ndarray, shape (N, D)
        Precomputed embeddings for N samples.
        (These should match the dimension & normalization of the FAISS index.)
    labels : np.ndarray, shape (N,)
        Class or superfamily labels for each sample.
    index : faiss.Index
        A FAISS index that already has the same embeddings added.
    k : int
        Number of nearest neighbors to query, and total negative samples to add per anchor.
    num_triplets : int
        Number of anchor-positive picks (each yields up to k negative triplets).

    Returns
    -------
    triplets : list of tuples
        Each tuple is (anchor_idx, positive_idx, negative_idx).
        Note that each anchor/positive pair can yield multiple negative samples.
    """
    triplets = []

    # Build label -> indices dictionary for picking positives.
    label_dict = {}
    for i, lbl in enumerate(labels):
        if lbl not in label_dict:
            label_dict[lbl] = []
        label_dict[lbl].append(i)

    N = len(labels)

    for _ in tqdm(range(num_triplets), desc=f"Generating triplets"):
        # 1) Random anchor
        anchor_idx = np.random.randint(0, N)
        anchor_label = labels[anchor_idx]
        anchor_emb = embeddings[anchor_idx]

        # 2) Positive from same label
        positive_idx = np.random.choice(label_dict[anchor_label])

        # 3) Query FAISS for top-k neighbors
        #    Reshape anchor_emb to [1, D] for FAISS
        _, I = index.search(anchor_emb.reshape(1, -1), k)
        neighbors = I[0]  # Indices of top-k neighbors

        # Identify mislabeled neighbors (i.e., different label than anchor)
        mislabeled = [neg for neg in neighbors if labels[neg] != anchor_label]
        mislabeled_count = len(mislabeled)

        # Add triplets for mislabeled neighbors
        for neg_idx in mislabeled:
            triplets.append((anchor_idx, positive_idx, neg_idx))

        # If we haven't reached k negatives via mislabel, pick random negatives until we hit k
        needed = k - mislabeled_count
        if needed > 0:
            # All possible negative candidates = indices with different label
            negative_candidates = np.where(labels != anchor_label)[0]
            # Randomly select 'needed' distinct negatives
            random_negatives = np.random.choice(negative_candidates, size=needed, replace=False)
            for neg_idx in random_negatives:
                triplets.append((anchor_idx, positive_idx, neg_idx))

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
def train_projection_head(train_embeddings, train_labels, test_embeddings, test_labels, train_index, test_index, epochs=10, batch_size=256, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=128, normalize_output=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=0.2)

    train_dataset = ProteinDataset(train_embeddings, train_labels)
    test_dataset = ProteinDataset(test_embeddings, test_labels)

    for epoch in range(epochs):

        train_triplets = get_triplets_faiss(train_embeddings, train_labels, train_index, k=25, num_triplets=5000)
        test_triplets = get_triplets_faiss(test_embeddings, test_labels, test_index, k=25, num_triplets=1000)

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

    return model

# Main execution
if __name__ == "__main__":

    print("Fetching train-test split...")
    train_embeddings, test_embeddings, train_labels, test_labels = fetch_train_test_split()

    train_index = faiss.read_index('/scratch/gpfs/jr8867/embeddings/scop/train-test-split/protein_embeddings_train.index')
    test_index = faiss.read_index('/scratch/gpfs/jr8867/embeddings/scop/train-test-split/protein_embeddings_test.index')
    print("Starting training...")
    
    # Train the model
    projection_model = train_projection_head(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        train_index, test_index,
        epochs=25, batch_size=256, lr=0.001
    )

    # Save model
    output_dir = "/scratch/gpfs/jr8867/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "projection_model_mining.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

