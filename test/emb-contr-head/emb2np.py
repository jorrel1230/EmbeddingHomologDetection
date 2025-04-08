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

def process_and_save_embeddings(df, output_dir, batch_size=16, resume_from=0):
    """
    Process sequences in batches and save to .npy files.
    Includes resume capability.
    
    Args:
    - df (DataFrame): DataFrame containing sequences and metadata
    - output_dir (str): Directory to save numpy arrays
    - batch_size (int): Number of sequences to process in each batch
    - resume_from (int): Index to resume processing from
    
    Returns:
    - total_processed: Number of sequences processed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches
    total_processed = resume_from
    batch_num = resume_from // batch_size
    
    for i in tqdm(range(resume_from, len(df), batch_size), ncols=100):
        batch_df = df.iloc[i:i+batch_size]
        batch_indices = batch_df['index'].values
        
        try:
            # Get embeddings for this batch
            batch_embeddings = embed_sequence_batch(batch_df['seq'].tolist())
            
            # Save this batch to a .npy file
            embeddings_file = os.path.join(output_dir, f'batch-{batch_num}-embeddings.npy')
            indices_file = os.path.join(output_dir, f'batch-{batch_num}-indices.npy')
            
            np.save(embeddings_file, batch_embeddings.astype('float32'))
            np.save(indices_file, batch_indices.astype('int64'))
            
            print(f"Saved batch {batch_num} with {len(batch_embeddings)} embeddings")
            
            total_processed += len(batch_embeddings)
            batch_num += 1
            
            # Save a checkpoint file with the current progress
            with open(os.path.join(output_dir, 'checkpoint.txt'), 'w') as f:
                f.write(str(total_processed))
        
        except Exception as e:
            print(f"Error processing batch at index {i}: {str(e)}")
            
            # Save checkpoint
            with open(os.path.join(output_dir, 'checkpoint.txt'), 'w') as f:
                f.write(str(total_processed))
    
    return total_processed

if __name__ == "__main__":
    
    # Load data
    scop_csv_path = '/scratch/gpfs/jr8867/datasets/scop/scop_data.csv'
    scop_df = pd.read_csv(scop_csv_path)
    
    # Set parameters
    output_dir = '/scratch/gpfs/jr8867/embeddings/scop'
    batch_size = 5000
    
    # Check if we need to resume from a checkpoint
    checkpoint_file = os.path.join(output_dir, 'checkpoint.txt')
    resume_from = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            resume_from = int(f.read().strip())
        print(f"Resuming from checkpoint: {resume_from} embeddings already processed")
    
    # Process sequences
    total_processed = process_and_save_embeddings(scop_df, output_dir, batch_size, resume_from)
    print(f"Completed processing {total_processed} sequences")
