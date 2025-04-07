import torch
import torch.nn as nn
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import gc
import os

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
device = torch.device("cuda" if cuda_available else "cpu")

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def search_similar_proteins(query_embedding, index, superfamilies, families, k=5):
    # Search in FAISS index
    D, I = index.search(query_embedding, k)
    
    # Get the indices and distances for the first query
    neighbor_indices = I[0]
    neighbor_distances = D[0]
    
    # Get superfamilies and families for each neighbor
    neighbor_superfamilies = superfamilies[neighbor_indices]
    neighbor_families = families[neighbor_indices]
    
    # Return a list of tuples containing (index, distance, superfamily, family)
    results = []
    for idx, dist, sf, fa in zip(neighbor_indices, neighbor_distances, neighbor_superfamilies, neighbor_families):
        results.append({
            'index': idx,
            'distance': dist,
            'superfamily': sf,
            'family': fa
        })
        
    return results

def evaluate_similarity_search(index, embeddings, indicies, superfamilies, families, k=5, batch_size=10):
    """
    Run similarity search on all sequences and collect similarity scores + ground truth labels.
    Process in small batches to avoid memory issues.
    
    Returns:
    - all_scores: Numpy array of similarity scores
    - all_labels: Numpy array of ground truth labels (1 = homolog, 0 = non-homolog)
    """
    all_scores = []
    all_labels = []

    num_sequences = len(embeddings)
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    print(f"Processing {num_sequences} sequences in {num_batches} batches...")
    
    for batch_idx in tqdm(range(0, num_sequences, batch_size)):
        # Clear GPU cache and system memory before processing each batch
        if cuda_available: torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(batch_idx + batch_size, num_sequences)
        # print(f"Processing batch {batch_idx//batch_size + 1}/{num_batches} (sequences {batch_idx}-{batch_end-1})")
        
        for i in range(batch_idx, batch_end):
            # Get the query embedding
            query_embedding = embeddings[i].reshape(1, -1)
            query_superfamily = superfamilies[i]
            query_family = families[i]
            
            # Search for similar proteins
            results = search_similar_proteins(query_embedding, index, superfamilies, families, k=k)
            
            # Process each result
            for result in results:
                neighbor_superfamily = result['superfamily']
                neighbor_family = result['family']
                distance = result['distance']
                
                # Determine if this is a homolog (same family or superfamily)
                is_homolog = 0
                if query_family == neighbor_family or query_superfamily == neighbor_superfamily:
                    is_homolog = 1
                
                # Add to our results
                all_scores.append(distance)
                all_labels.append(is_homolog)

    return np.array(all_scores), np.array(all_labels)

def save_results(all_scores, all_labels, output_dir, set_type):
    """Save evaluation results to disk"""


    if set_type == 'train':
        output_dir = os.path.join(output_dir, 'train')
    elif set_type == 'test':
        output_dir = os.path.join(output_dir, 'test')
    else:
        raise ValueError(f"Invalid type: {set_type}")

    os.makedirs(output_dir, exist_ok=True)
    
    # Save scores and labels
    np.save(f'{output_dir}/{set_type}_scores.npy', all_scores)
    np.save(f'{output_dir}/{set_type}_labels.npy', all_labels)
    
    # Compute and save ROC curve
    # Invert scores since smaller distances indicate higher similarity
    fpr, tpr, thresholds = roc_curve(all_labels, -all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Save ROC data
    np.savez(f'{output_dir}/{set_type}_roc_data.npz', 
             fpr=fpr, tpr=tpr, thresholds=thresholds, auc=roc_auc)
    
    # Plot ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Refined Embeddings')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/{set_type}_roc_curve.png')
    plt.close()
    
    print(f"Results saved to {output_dir}")
    print(f"AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    # Set k for nearest neighbors search
    k_neighbors = 100  # Adjust based on memory availability

    # Train set
    train_index = faiss.read_index('/scratch/gpfs/jr8867/main/db/train-test-fold/train_embeddings.index')
    train_embeddings = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/train_embeddings.npy')
    train_indices = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/train_indicies.npy')
    train_superfamilies = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/train_superfamilies.npy')
    train_families = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/train_families.npy')
    
    train_scores, train_labels = evaluate_similarity_search(
        train_index, train_embeddings, train_indices, train_superfamilies, train_families, 
        k=k_neighbors 
    )

    # Save results instead of plotting in-memory
    save_results(train_scores, train_labels, '/scratch/gpfs/jr8867/main/baseline/evals', 'train')

    del train_index, train_embeddings, train_indices, train_superfamilies, train_families
    gc.collect()
    
    # Test set
    test_index = faiss.read_index('/scratch/gpfs/jr8867/main/db/train-test-fold/test_embeddings.index')
    test_embeddings = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/test_embeddings.npy')
    test_indices = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/test_indicies.npy')
    test_superfamilies = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/test_superfamilies.npy')
    test_families = np.load('/scratch/gpfs/jr8867/main/db/train-test-fold/test_families.npy')

    test_scores, test_labels = evaluate_similarity_search(
        test_index, test_embeddings, test_indices, test_superfamilies, test_families, 
        k=k_neighbors 
    )
    save_results(test_scores, test_labels, '/scratch/gpfs/jr8867/main/baseline/evals', 'test')

    del test_index, test_embeddings, test_indices, test_superfamilies, test_families
    gc.collect()