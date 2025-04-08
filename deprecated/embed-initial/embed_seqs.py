import os
import itertools
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import faiss
import esm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from Bio import SeqIO

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
device = torch.device("cuda" if cuda_available else "cpu")

# Load model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.to(device)
model.eval()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def embed_sequence_batch(sequences, batch_size=8, max_length=1022):
    """
    Convert a batch of protein sequences to L2-normalized embeddings using ESM-2 with mean pooling.
    Includes error handling and sequence length limiting.
    
    Args:
    - sequences (list): List of protein sequences as strings.
    - batch_size (int): Not used, kept for compatibility.
    - max_length (int): Maximum sequence length to process.
    
    Returns:
    - embeddings (np.array): Array of shape (N, D) where N is number of sequences, D is embedding size.
    """
    all_embeddings = []
    sub_batch_size = 1
    
    for i in range(0, len(sequences), sub_batch_size):
        sub_batch_seqs = sequences[i:i+sub_batch_size]
        
        # Truncate sequences that are too long
        truncated_seqs = [seq[:max_length] for seq in sub_batch_seqs]
        sub_batch_data = [(str(j), seq) for j, seq in enumerate(truncated_seqs)]
        
        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(sub_batch_data)
            batch_tokens = batch_tokens.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            
            token_embeddings = results["representations"][33]
            
            for j in range(len(truncated_seqs)):
                valid_tokens = batch_tokens[j] != alphabet.padding_idx
                seq_embedding = token_embeddings[j, valid_tokens].mean(dim=0).cpu().numpy()
                
                # L2-normalize the embedding
                seq_embedding = normalize(seq_embedding.reshape(1, -1)).flatten()
                
                all_embeddings.append(seq_embedding)
            
        except Exception as e:
            print(f"Error processing sequence at index {i}: {str(e)}")
            # Add a zero embedding as a placeholder for failed sequences
            zero_embedding = np.zeros(1280, dtype=np.float32)
            all_embeddings.append(zero_embedding)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.array(all_embeddings)

def process_and_save_embeddings(df, output_dir, batch_size=16, save_every=100, resume_from=0):
    """
    Process sequences in batches, update FAISS index, and save periodically.
    Includes resume capability.
    
    Args:
    - df (DataFrame): DataFrame containing sequences and metadata
    - output_dir (str): Directory to save FAISS index and metadata
    - batch_size (int): Number of sequences to process in each batch for FAISS updates
    - save_every (int): Save after processing this many sequences
    - resume_from (int): Index to resume processing from
    
    Returns:
    - index: Final FAISS index
    - metadata: DataFrame with metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup FAISS index
    dimension = 1280  # ESM-2 embedding dimension
    faiss_path = os.path.join(output_dir, 'protein_embeddings_norm.index')
    
    # Check if we need to resume from an existing index
    if resume_from > 0 and os.path.exists(faiss_path):
        print(f"Resuming from existing index with {resume_from} embeddings")
        index = faiss.read_index(faiss_path)
    else:
        print("Creating new index")
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        resume_from = 0
    
    # Process in batches
    total_processed = resume_from
    
    for i in tqdm(range(resume_from, len(df), batch_size), ncols=100):
        batch_df = df.iloc[i:i+batch_size]
        
        try:
            # Get embeddings for this batch
            batch_embeddings = embed_sequence_batch(batch_df['seq'].tolist())
            
            # Get the original indices from the DataFrame
            batch_indices = batch_df['index'].values
            
            # Add to FAISS index
            index.add_with_ids(
                batch_embeddings.astype('float32'),
                np.array(batch_indices, dtype=np.int64)
            )
            
            total_processed += len(batch_embeddings)
            
            # Save periodically
            if total_processed % save_every == 0 or i + batch_size >= len(df):
                faiss.write_index(index, faiss_path)
                print(f"Saved {total_processed} embeddings to {faiss_path}")
                
                # Also save a checkpoint file with the current progress
                with open(os.path.join(output_dir, 'checkpoint.txt'), 'w') as f:
                    f.write(str(total_processed))
        
        except Exception as e:
            print(f"Error processing batch at index {i}: {str(e)}")
            # Save the current progress before exiting
            faiss.write_index(index, faiss_path)
            print(f"Saved {total_processed} embeddings to {faiss_path} before error")
            
            # Save checkpoint
            with open(os.path.join(output_dir, 'checkpoint.txt'), 'w') as f:
                f.write(str(total_processed))
    
    return index, df

if __name__ == "__main__":
    
    # Load data
    scop_csv_path = '/scratch/gpfs/jr8867/datasets/scop/scop_data.csv'
    scop_df = pd.read_csv(scop_csv_path)
    
    # Set parameters
    output_dir = '/scratch/gpfs/jr8867/embeddings/scop'
    batch_size = 16
    save_every = 100
    
    # Check if we need to resume from a checkpoint
    checkpoint_file = os.path.join(output_dir, 'checkpoint.txt')
    resume_from = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            resume_from = int(f.read().strip())
        print(f"Resuming from checkpoint: {resume_from} embeddings already processed")
    
    # Process sequences
    index, metadata = process_and_save_embeddings(scop_df, output_dir, batch_size, save_every, resume_from)
