import os
import torch
import esm
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

# Argument parsing for SLURM input file
parser = argparse.ArgumentParser(description="Process mutation data with ESM1v.")
parser.add_argument("--input", required=True, help="Path to the input CSV file")
parser.add_argument("--output", required=True, help="Path to save the output CSV file")
args = parser.parse_args()

INPUT_FILE = args.input  # File passed via SLURM job
OUTPUT_FILE = args.output
# Path where ESM models are cached
cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints/")
model_filename = "esm1v_t33_650M_UR90S_1.pt"
model_path = os.path.join(cache_dir, model_filename)

# Load ESM-1v Model
print(f"Checking for ESM-1v Model on Node {os.getenv('SLURM_NODEID', 'Unknown')}...")
if not os.path.exists(model_path):
    print("Model not found in cache. Downloading...")
else:
    print("Model found in cache. Loading without download.")

model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
batch_converter = alphabet.get_batch_converter()
model.eval()  # Set model to evaluation mode

# Load Dataset
print(f"Loading Dataset: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# **Preprocessing Step: Remove rows where mutation position is missing**
df = df.dropna(subset=['pos_in_uniprot'])
df['pos_in_uniprot'] = df['pos_in_uniprot'].astype(int)  # Ensure mutation position is integer

# Filter proteins with sequence length > 1024
protein_sequences = df[['uniprot_ID', 'Protein Sequence']].copy()
protein_dict = dict(zip(protein_sequences['uniprot_ID'], protein_sequences['Protein Sequence']))
df_filtered = df[df['uniprot_ID'].map(lambda pid: len(protein_dict[pid]) > 1024)].copy()

# **FOR INITIAL TESTING: Limit to first 5 rows per file** (REMOVE LATER FOR FULL BATCH PROCESSING)
#df_filtered = df_filtered.head(5)

# Sliding window parameters
WINDOW_SIZE = 1000  # Length of sequence chunk
STEP_SIZE = 250  # Overlapping step

# Function to extract sliding windows
def extract_windows(sequence, mutation_pos, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Generates multiple overlapping windows if the sequence is >1024 residues."""
    seq_len = len(sequence)
    windows = []
    adjusted_positions = []

    # If the sequence is short, use full-length sequence without sliding
    if seq_len <= window_size:
        return [(sequence, mutation_pos - 1)]  # 1-based to 0-based index

    # Generate overlapping windows
    for start in range(0, seq_len - window_size + 1, step_size):
        end = start + window_size
        if mutation_pos >= start and mutation_pos < end:  # Ensure mutation falls in window
            adjusted_mut_pos = (mutation_pos - 1) - start  # Fixed indexing
            windows.append(sequence[start:end])
            adjusted_positions.append(adjusted_mut_pos)

    return list(zip(windows, adjusted_positions))  # Return list of (window, adj_pos)


# Function to Compute Log-Likelihood Ratio Scores
def compute_mutation_effect(protein_id, mutation_position, wildtype_aa, mutant_aa, mutated_sequence):
    """Compute zero-shot mutation effect using ESM-1v"""

    # Ensure protein exists in dictionary
    if protein_id not in protein_dict:
        print(f"Error: Protein ID {protein_id} not found in dictionary!")
        return None

    wildtype_seq = protein_dict[protein_id]

    # Ensure mutation position is valid
    if pd.isna(mutation_position) or not isinstance(mutation_position, int):
        print(f"Error: Invalid mutation position: {mutation_position}")
        return None

    if mutation_position > len(wildtype_seq):
        print(f" Error: Mutation position {mutation_position} is out of bounds (Seq Len: {len(wildtype_seq)})")
        return None

    # Debugging: Print Original Sequence Length
    print(f"Original Sequence Length for {protein_id}: {len(wildtype_seq)}")

    # Extract sliding windows
    wildtype_windows = extract_windows(wildtype_seq, mutation_position)
    mutated_windows = extract_windows(mutated_sequence, mutation_position)

    # Ensure mutation appears in at least one window
    if not wildtype_windows or not mutated_windows:
        print(f"Warning: Mutation position {mutation_position} not in any valid window!")
        return None

    mutation_scores = []  # Store all predictions from different windows

    for (wildtype_window, adj_mut_pos), (mutated_window, _) in zip(wildtype_windows, mutated_windows):
        # **Check if wild-type amino acid matches expected at adjusted position**
        if wildtype_window[adj_mut_pos] != wildtype_aa:
            print(f"Error: WT amino acid mismatch at adjusted position {adj_mut_pos + 1}."
                  f" Expected: {wildtype_aa}, Found: {wildtype_window[adj_mut_pos]}")
            continue  # Skip this window

        # **Check if mutated amino acid is correctly applied**
        if mutated_window[adj_mut_pos] != mutant_aa:
            print(f"Error: Mutated amino acid mismatch at adjusted position {adj_mut_pos + 1}."
                  f" Expected: {mutant_aa}, Found: {mutated_window[adj_mut_pos]}")
            continue  # Skip this window

        # Debugging: Print Mutation Position Before and After Sliding Window
        print(f"Processing Mutation | Protein ID: {protein_id}, Position: {mutation_position}, "
              f"Adjusted Pos: {adj_mut_pos + 1}, WT_AA: {wildtype_aa}, MUT_AA: {mutant_aa}")

        # Tokenize sequences
        data = [("wildtype", wildtype_window), ("mutant", mutated_window)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            logits = model(batch_tokens, repr_layers=[33])["logits"]

        # Compute log-likelihood for adjusted mutation position
        wildtype_logits = logits[0, adj_mut_pos, :]
        mutant_logits = logits[1, adj_mut_pos, :]

        # Mutation effect score (log-likelihood ratio)
        mutation_score = torch.log_softmax(wildtype_logits, dim=-1)[alphabet.get_idx(wildtype_aa)] - \
                         torch.log_softmax(mutant_logits, dim=-1)[alphabet.get_idx(mutant_aa)]

        mutation_scores.append(mutation_score.item())

    # If no valid scores, return None
    if not mutation_scores:
        return None

    # Print all individual scores before averaging
    print(f"Individual scores for {protein_id}, Mutation Position {mutation_position}: {mutation_scores}")

    # Aggregate multiple scores for the same mutation
    final_score = np.mean(mutation_scores)  # Use mean to aggregate
    return final_score


# **Processing Mutations for Each File in Parallel Across Nodes**
print(f"Processing first 5 mutations from {INPUT_FILE} on {os.getenv('SLURM_NODEID', 'Unknown')}...")
mutation_scores = []
for i, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    score = compute_mutation_effect(
        row['uniprot_ID'],
        row['pos_in_uniprot'],
        row['wt_aa'],
        row['mut_aa'],
        row['Mutated_Protein_Sequence']
    )
    mutation_scores.append(score)

# Save results
df_filtered['Mutation_Score'] = mutation_scores
df_filtered.to_csv(OUTPUT_FILE, index=False)

# Print results
print(f"\nProcessing Complete! Results saved to {OUTPUT_FILE} on Node {os.getenv('SLURM_NODEID', 'Unknown')}.")
