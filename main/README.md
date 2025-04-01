# Main Pipeline for Embedding Homolog Detection

## Overview
This pipeline processes sequences from the SCOP dataset to generate embeddings, which are then refined and stored in a vector database for querying homologs.

## Steps

1. **Data Acquisition**: 
   - The pipeline begins by loading sequences from the SCOP dataset.

2. **Embedding Generation**:
   - Sequences are fed into the ESM2 model to generate embeddings. 
   - A mean pooling operation is applied to obtain fixed-size embeddings from the variable-length sequences.

3. **Projection Head**:
   - The generated embeddings are passed through a Projection Head, which is a Multi-Layer Perceptron (MLP) trained using contrastive learning with triplet loss. 
   - This step tunes the embeddings to enhance their discriminative power.

4. **Vector Database Storage**:
   - The refined embeddings are stored in a vector database, allowing for efficient querying of homologous sequences.

5. **Querying for Homologs**:
   - Users can query the vector database to find homologs based on the stored embeddings.

## Training the MLP
- During the training of the MLP, hard negatives are identified to generate effective triplets. This process helps in improving the model's ability to distinguish between similar and dissimilar sequences.
