import torch
import torch.nn as nn
import numpy as np
import faiss

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Projection Head Model
# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim, output_dim=256):
#         super(ProjectionHead, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LeakyReLU(0.1),
#             nn.Linear(512, output_dim),
#         )
    
#     def forward(self, x):
#         return self.model(x)


# Load the saved model
# projection_model = ProjectionHead(input_dim=1280, output_dim=256)
# projection_model.load_state_dict(torch.load("/scratch/gpfs/jr8867/main/contr-v1/models/contr-v1-large.pth"))
# projection_model.to(device)
# projection_model.eval()

# Fetching Data from Storages
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

print("Fetching data...")

train_embeddings, train_indicies, train_superfamilies, train_families, train_folds = fetch_data('train')
test_embeddings, test_indicies, test_superfamilies, test_families, test_folds = fetch_data('test')

print(f"Train embeddings shape: {train_embeddings.shape}")
print(f"Train indicies shape: {train_indicies.shape}")
print(f"Train superfamilies shape: {train_superfamilies.shape}")
print(f"Train families shape: {train_families.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")
print(f"Test indicies shape: {test_indicies.shape}")
print(f"Test superfamilies shape: {test_superfamilies.shape}")
print(f"Test families shape: {test_families.shape}")

# Save new embeddings
# with torch.no_grad():
#     train_projected_embeddings = projection_model(torch.tensor(train_embeddings, dtype=torch.float32).to(device)).cpu().numpy()
#     test_projected_embeddings = projection_model(torch.tensor(test_embeddings, dtype=torch.float32).to(device)).cpu().numpy()

# print(f"Train Refined embeddings shape: {train_projected_embeddings.shape}")
# print(f"Test Refined embeddings shape: {test_projected_embeddings.shape}")

# np.save("/scratch/gpfs/jr8867/main/contr-v1/embeddings/train_projected_embeddings.npy", train_projected_embeddings)
# np.save("/scratch/gpfs/jr8867/main/contr-v1/embeddings/test_projected_embeddings.npy", test_projected_embeddings)

# FAISS Indexing
train_index = faiss.IndexFlatL2(train_embeddings.shape[1])
train_index.add(train_embeddings)
faiss.write_index(train_index, "/scratch/gpfs/jr8867/main/db/train-test-fold/train_embeddings.index")

test_index = faiss.IndexFlatL2(test_embeddings.shape[1])
test_index.add(test_embeddings)
faiss.write_index(test_index, "/scratch/gpfs/jr8867/main/db/train-test-fold/test_embeddings.index")






print("Saved refined embeddings and FAISS index!")
