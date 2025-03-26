import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

# Define MLP Projection Head
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


# Load the saved model
projection_model = ProjectionHead(input_dim=1280, output_dim=128)
projection_model.load_state_dict(torch.load("/scratch/gpfs/jr8867/models/projection_model_mining.pth"))
projection_model.to("cuda")
projection_model.eval()

# Fetching Data from Storages
def fetch_split(split_type):
    if split_type == "train":
        X = np.load('/scratch/gpfs/jr8867/embeddings/scop/train-test-split/train_embeddings.npy')
        y = np.load('/scratch/gpfs/jr8867/embeddings/scop/train-test-split/train_labels.npy')
    elif split_type == "test":
        X = np.load('/scratch/gpfs/jr8867/embeddings/scop/train-test-split/test_embeddings.npy')
        y = np.load('/scratch/gpfs/jr8867/embeddings/scop/train-test-split/test_labels.npy')
    else:
        raise ValueError(f"Invalid split type: {split_type}")
    
    return X, y

# Load data
print("Loading embeddings and indices...")
X_train, y_train = fetch_split("train")
X_test, y_test = fetch_split("test")

print(f"Train embeddings shape: {X_train.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Test embeddings shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Transform ESM embeddings using projection head
with torch.no_grad():
    y_train_proj = projection_model(torch.tensor(X_train, dtype=torch.float32).to("cuda")).cpu().numpy()
    np.save("/scratch/gpfs/jr8867/embeddings/scop/train-test-split/train_embeddings_proj.npy", y_train_proj)


# FAISS Indexing
index = faiss.IndexFlatL2(y_train_proj.shape[1])
index.add(y_train_proj)
faiss.write_index(index, "/scratch/gpfs/jr8867/embeddings/scop/train-test-split/train_embeddings_proj_L2.index")

print("Saved train embeddings and FAISS index!")

with torch.no_grad():
    y_test_proj = projection_model(torch.tensor(X_test, dtype=torch.float32).to("cuda")).cpu().numpy()
    np.save("/scratch/gpfs/jr8867/embeddings/scop/train-test-split/test_embeddings_proj.npy", y_test_proj)

# FAISS Indexing
index = faiss.IndexFlatL2(y_test_proj.shape[1])
index.add(y_test_proj)
faiss.write_index(index, "/scratch/gpfs/jr8867/embeddings/scop/train-test-split/test_embeddings_proj_L2.index")

print("Saved test embeddings and FAISS index!")
