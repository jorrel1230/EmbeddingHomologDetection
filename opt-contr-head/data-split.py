#!/usr/bin/env python

import os
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

save_dir = '/scratch/gpfs/jr8867/embeddings/scop/train-test-split'

# if dir doesn't exist, create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(f'{save_dir}/train_embeddings.npy', train_embeddings)
np.save(f'{save_dir}/test_embeddings.npy', test_embeddings)
np.save(f'{save_dir}/train_labels.npy', train_labels)
np.save(f'{save_dir}/test_labels.npy', test_labels)