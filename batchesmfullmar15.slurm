#!/bin/bash
#SBATCH --job-name=mutated_array       # Job name
#SBATCH --output=batchesmfullmar15/batchesm2_%A_%a.log  # Output file per task
#SBATCH --error=batchesmfullmar15/batchesm2_%A_%a.err   # Error file per task
#SBATCH --time=0-00:00:00              # Infinite runtime
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4              # Allocate 4 CPU cores
#SBATCH --mem=32G                      # Memory allocation
#SBATCH --ntasks=1                     # Run as a single task
#SBATCH --array=1-21                    # Define **21** tasks

# Create the directory if it doesn't exist
mkdir -p /scratch/rdeo/ESM1vZeroshot/batchesmfullmar15

# Load the environment
source /scratch/rdeo/ESM1vZeroshot/esm1v_env/bin/activate

# Extract the input file name from config
INPUTFILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/rdeo/ESM1vZeroshot/run.config)

# Check if the input file exists
if [ ! -f "$INPUTFILE" ]; then
    echo "Error: Input file $INPUTFILE does not exist!" > /scratch/rdeo/ESM1vZeroshot/batchesmfullmar15/batchesm2_${SLURM_ARRAY_TASK_ID}.err
    exit 1
fi

# Define the output directory for results
RESULTS_DIR="/scratch/rdeo/ESM1vZeroshot/batchesmfullmar15"
OUTPUT_FILE="${RESULTS_DIR}/$(basename "$INPUTFILE" .csv)_results.csv"

# Run the Python script and save results in the specified directory
python /scratch/rdeo/ESM1vZeroshot/batchesmfullmar15.py --input "$INPUTFILE" --output "$OUTPUT_FILE"
