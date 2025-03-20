# Function to search for similar proteins
import faiss
import numpy as np
import pandas as pd
import torch
import esm
from sklearn.preprocessing import normalize

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
device = torch.device("cuda" if cuda_available else "cpu")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.to(device)
model.eval() # Ensures that the model is in evaluation mode, not training mode

def embed_sequence(sequence):
    """
    Convert a protein sequence to L2-normalized embedding using ESM-2 with mean pooling.
    
    Args:
    - sequence (str): A protein sequence as string.

    Returns:
    - embedding (np.array): L2-normalized array of shape (D,) where D is embedding size.
    """

    # Tokenize sequence
    data = [(str(0), sequence)]
    _, _, batch_tokens = batch_converter(data)

    # Move to GPU if available
    batch_tokens = batch_tokens.to(device)

    # Forward pass to get embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Extract embedding (mean pooling excluding padding)
    token_embeddings = results["representations"][33]  # Extract last hidden layer (layer 33 for this model)
    
    # Apply mean pooling
    valid_tokens = batch_tokens[0] != alphabet.padding_idx  # Mask out padding tokens
    seq_embedding = token_embeddings[0, valid_tokens].mean(dim=0).cpu().numpy()
    
    # L2-normalize the embedding
    seq_embedding = normalize(seq_embedding.reshape(1, -1)).flatten()

    return seq_embedding.astype('float32')


def search_similar_proteins(query_seq, index, metadata, k=5):
    """
    Search for similar proteins using a query sequence.
    
    Args:
    - query_seq (str): Query protein sequence
    - index: FAISS index
    - metadata (DataFrame): Metadata DataFrame
    - k (int): Number of nearest neighbors to return
    
    Returns:
    - DataFrame: Metadata of similar proteins
    """
    # Embed query sequence
    query_embedding = embed_sequence(query_seq)
    query_embedding = np.array([query_embedding]).astype('float32')
    
    # Search in FAISS index
    D, I = index.search(query_embedding, k)
    
    # Get metadata for results - explicitly create a copy
    result_indices = I[0]
    similar_proteins = metadata[metadata['index'].isin(result_indices)].copy()
    
    # Add distance information
    distances = {idx: dist for idx, dist in zip(result_indices, D[0])}
    similar_proteins['distance'] = similar_proteins['index'].map(distances)
    
    # Sort by distance
    # For IndexFlatIP with normalized vectors, higher values are better (more similar)
    similar_proteins = similar_proteins.sort_values('distance', ascending=False)
    
    return similar_proteins


if __name__ == "__main__":

    scop_csv_path = '/scratch/gpfs/jr8867/datasets/scop/scop_data.csv'
    scop_df = pd.read_csv(scop_csv_path)

    # Use the normalized index
    index = faiss.read_index('/scratch/gpfs/jr8867/embeddings/scop/protein_embeddings_norm.index')

    for i in range(5):
        print(f'Query sequence {i}:')
        example_seq = scop_df.iloc[i]['seq']
        similar = search_similar_proteins(example_seq, index, scop_df, k=5)
        print("Similar proteins:")
        print(similar)
        print('-'*100)

    # print(scop_df.head())