import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import esm
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import gc
import os
import sys

# Get batch number from SLURM_ARRAY_TASK_ID environment variable
try:
    batch_num = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Using SLURM_ARRAY_TASK_ID: {batch_num}")
except (KeyError, ValueError):
    # If not running as a SLURM job array or invalid value, use command line argument or default
    if len(sys.argv) > 1:
        batch_num = int(sys.argv[1])
        print(f"Using command line argument for batch number: {batch_num}")
    else:
        batch_num = 0  # Default batch number
        print(f"Using default batch number: {batch_num}")

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
device = torch.device("cuda" if cuda_available else "cpu")

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# Load model only when needed
def get_model():
    # Load ESM model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    model.eval()
    
    # Load projection model
    projection_model = ProjectionHead(input_dim=1280, output_dim=128)
    projection_model.load_state_dict(torch.load("/scratch/gpfs/jr8867/models/projection_model_old.pth"))
    projection_model.to(device)
    projection_model.eval()
    
    return model, alphabet, projection_model

def embed_sequence(sequence, model=None, alphabet=None, projection_model=None, batch_converter=None, max_length=1022):
    """
    Convert a protein sequence to L2-normalized embedding using ESM-2 with mean pooling,
    then refine it with the projection head.
    
    Args:
    - sequence (str): A protein sequence as string.
    - model: ESM model (loaded on demand if None)
    - alphabet: ESM alphabet (loaded on demand if None)
    - projection_model: Projection head model (loaded on demand if None)
    - batch_converter: ESM batch converter (loaded on demand if None)
    - max_length: Maximum sequence length to process

    Returns:
    - embedding (np.array): L2-normalized array of shape (D,) where D is embedding size.
    """
    # Load models on demand if not provided
    if model is None or alphabet is None or projection_model is None or batch_converter is None:
        model, alphabet, projection_model = get_model()
        batch_converter = alphabet.get_batch_converter()
        temp_model = True
    else:
        temp_model = False

    try:
        # Truncate sequence if too long
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
            
        # Tokenize sequence
        data = [(str(0), sequence)]
        _, _, batch_tokens = batch_converter(data)

        # Move to GPU if available
        batch_tokens = batch_tokens.to(device)

        # Clear CUDA cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Forward pass to get embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        # Extract embedding (mean pooling excluding padding)
        token_embeddings = results["representations"][33]
        
        # Apply mean pooling
        valid_tokens = batch_tokens[0] != alphabet.padding_idx
        seq_embedding = token_embeddings[0, valid_tokens].mean(dim=0)
        
        # L2-normalize the embedding
        seq_embedding = F.normalize(seq_embedding, p=2, dim=0)
        
        # Put the sequence through the projection head
        with torch.no_grad():
            refined_embedding = projection_model(seq_embedding)
            
        return refined_embedding.cpu().numpy().astype('float32')
    
    finally:
        # Clean up if we created temporary models
        if temp_model:
            del model, alphabet, projection_model, batch_converter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

def search_similar_proteins(query_seq, index, metadata, k=5, model=None, alphabet=None, projection_model=None, batch_converter=None):
    """
    Search for similar proteins using a query sequence.
    
    Args:
    - query_seq (str): Query protein sequence
    - index: FAISS index
    - metadata (DataFrame): Metadata DataFrame
    - k (int): Number of nearest neighbors to return
    - model, alphabet, projection_model, batch_converter: Optional model components
    
    Returns:
    - DataFrame: Metadata of similar proteins
    """
    # Embed query sequence
    query_embedding = embed_sequence(query_seq, model, alphabet, projection_model, batch_converter)
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

def evaluate_similarity_search(index, metadata, start_idx, end_idx, k=5, batch_size=10):
    """
    Run similarity search on all sequences and collect similarity scores + ground truth labels.
    Process in small batches to avoid memory issues.
    
    Returns:
    - all_scores: Numpy array of similarity scores
    - all_labels: Numpy array of ground truth labels (1 = homolog, 0 = non-homolog)
    """
    all_scores = []
    all_labels = []
    
    # Load models once for all evaluations
    model, alphabet, projection_model = get_model()
    batch_converter = alphabet.get_batch_converter()

    # Clear GPU cache and system memory before processing each batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Process in small batches to avoid memory issues
        for i in tqdm(range(start_idx, end_idx, batch_size), desc='Evaluating Similarity Search', ncols=100):
            batch_end = min(i + batch_size, end_idx)
            
            for j in range(i, batch_end):
                query_seq = metadata.iloc[j]['seq']
                query_family = metadata.iloc[j]['sf']  # Ground truth family label

                # Search for similar proteins
                similar_proteins = search_similar_proteins(
                    query_seq, index, metadata, k, 
                    model, alphabet, projection_model, batch_converter
                )

                # Compare ground truth with retrieved sequences
                for _, row in similar_proteins.iterrows():
                    # Skip if the sequence is the same as the query sequence
                    if row['seq'] == query_seq:
                        continue
                    
                    score = row['distance']  # FAISS similarity distance
                    retrieved_family = row['sf']  # Retrieved protein's family

                    label = 1 if retrieved_family == query_family else 0  # Homolog or not
                    all_scores.append(score)
                    all_labels.append(label)
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    finally:
        # Clean up
        del model, alphabet, projection_model, batch_converter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return np.array(all_scores), np.array(all_labels)

def save_results(all_scores, all_labels, batch_num, total_batches, output_dir):
    """Save evaluation results to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scores and labels
    np.save(f'{output_dir}/scores_batch_{batch_num}_{total_batches}.npy', all_scores)
    np.save(f'{output_dir}/labels_batch_{batch_num}_{total_batches}.npy', all_labels)
    
    # Compute and save ROC curve
    # Invert scores since smaller distances indicate higher similarity
    fpr, tpr, thresholds = roc_curve(all_labels, -all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Save ROC data
    np.savez(f'{output_dir}/roc_data_batch_{batch_num}_{total_batches}.npz', 
             fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
    
    # Plot ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Refined Embeddings (Batch {batch_num})')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve_batch_{batch_num}_{total_batches}.png')
    plt.close()
    
    print(f"Results saved to {output_dir}")
    print(f"AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    # Set smaller batch size for processing
    processing_batch_size = 10 
    
    scop_csv_path = '/scratch/gpfs/jr8867/datasets/scop/scop_data.csv'
    scop_df = pd.read_csv(scop_csv_path)

    dataset_size = len(scop_df)
    num_ranges = 5

    range_size = dataset_size // num_ranges
    
    # Calculate start and end indices for the specific batch
    start_idx = batch_num * range_size
    end_idx = start_idx + range_size if batch_num < num_ranges - 1 else dataset_size

    print(f"Processing batch {batch_num}: sequences {start_idx} to {end_idx} (total: {end_idx-start_idx})")
    
    # Use the refined embeddings index
    index = faiss.read_index('/scratch/gpfs/jr8867/embeddings/scop/protein_embeddings_refined_old.index')

    # Set k for nearest neighbors search
    k_neighbors = 1000  # Adjust based on memory availability
    
    # Create output directory specific to this batch
    output_dir = f'/scratch/gpfs/jr8867/evals/protein-embeddings-refined/batch-{batch_num}-{num_ranges}'
    
    all_scores, all_labels = evaluate_similarity_search(
        index, scop_df, start_idx, end_idx, 
        k=k_neighbors, 
        batch_size=processing_batch_size
    )

    # Save results instead of plotting in-memory
    save_results(all_scores, all_labels, batch_num, num_ranges, output_dir)