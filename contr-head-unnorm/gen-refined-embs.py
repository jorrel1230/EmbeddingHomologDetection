import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

# Define MLP Projection Head
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


# Load the saved model
projection_model = ProjectionHead(input_dim=1280, output_dim=128)
projection_model.load_state_dict(torch.load("/scratch/gpfs/jr8867/models/projection_model_200k_best.pth"))
projection_model.to("cuda")
projection_model.eval()

# Fetching Data from Storages
def fetch_full_arr(arr_type):
    return np.load(f'/scratch/gpfs/jr8867/embeddings/scop/{arr_type}.npy')

# Load data
print("Loading embeddings and indices...")
embeddings = fetch_full_arr('embeddings')
indices = fetch_full_arr('indices')
print(f"Embeddings shape: {embeddings.shape}")
print(f"Indices shape: {indices.shape}")

# Save new embeddings
with torch.no_grad():
    refined_embeddings = projection_model(torch.tensor(embeddings, dtype=torch.float32).to("cuda")).cpu().numpy()

print(f"Refined embeddings shape: {refined_embeddings.shape}")

np.save("/scratch/gpfs/jr8867/embeddings/protein-embeddings-contrastive-unnorm/transformed_embeddings.npy", refined_embeddings)
np.save("/scratch/gpfs/jr8867/embeddings/protein-embeddings-contrastive-unnorm/indices.npy", indices)

# FAISS Indexing
index = faiss.IndexFlatL2(refined_embeddings.shape[1])
index.add(refined_embeddings)
faiss.write_index(index, "/scratch/gpfs/jr8867/embeddings/protein-embeddings-contrastive-unnorm/transformed_embeddings.index")

print("Saved refined embeddings and FAISS index!")
