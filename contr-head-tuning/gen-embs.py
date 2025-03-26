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
projection_model.load_state_dict(torch.load("/scratch/gpfs/jr8867/models/projection_model_finetuned.pth"))
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

np.save("/scratch/gpfs/jr8867/embeddings/scop/finetuned_embeddings.npy", refined_embeddings)

# FAISS Indexing
index = faiss.IndexFlatL2(refined_embeddings.shape[1])
index.add(refined_embeddings)
faiss.write_index(index, "/scratch/gpfs/jr8867/embeddings/scop/protein_embeddings_finetuned.index")

print("Saved refined embeddings and FAISS index!")
