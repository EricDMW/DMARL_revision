#!/bin/bash

# ============================================================================
# Multi-Seed Training Script for NMARL
# ============================================================================
# This script runs NMARL training with multiple random seeds and aggregates results
# for statistical analysis and publication-quality plots
# ============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Base directory for results
BASE_DIR="${SCRIPT_DIR}/results"
mkdir -p "${BASE_DIR}"

# Generate experiment ID (timestamp-based)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ID="seeds_${TIMESTAMP}"
RESULTS_DIR="${BASE_DIR}/${EXP_ID}"
mkdir -p "${RESULTS_DIR}"

# Default parameters
GRID_X=5
GRID_Y=5
MAX_EPISODES=200
STEPS=30
DDL=2
KAPPA_O=1
KAPPA_R=0
GAMMA=0.9
ETA=0.1

# Seeds to run (adjust as needed)
# Default: 10 seeds for statistical significance
SEEDS=(42 123 456 789 1000 2000 3000 4000 5000 6000)

# You can customize seeds by modifying the array above
# Example for 5 seeds: SEEDS=(42 123 456 789 1000)

# Number of seeds
N_SEEDS=${#SEEDS[@]}

echo "============================================================================"
echo "Multi-Seed Training Experiment (NMARL)"
echo "============================================================================"
echo "Experiment ID: ${EXP_ID}"
echo "Number of seeds: ${N_SEEDS}"
echo "Seeds: ${SEEDS[@]}"
echo "Results directory: ${RESULTS_DIR}"
echo "============================================================================"
echo ""

# Run training for each seed
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    SEED_DIR="${RESULTS_DIR}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    
    echo "============================================================================"
    echo "Running seed ${SEED} ($((i+1))/${N_SEEDS})"
    echo "============================================================================"
    
    # Note: stdout not redirected to allow tqdm progress bars to display
    python main.py \
        --grid-x ${GRID_X} \
        --grid-y ${GRID_Y} \
        --episodes ${MAX_EPISODES} \
        --steps ${STEPS} \
        --kappa-o ${KAPPA_O} \
        --kappa-r ${KAPPA_R} \
        --gamma ${GAMMA} \
        --eta ${ETA} \
        --seed ${SEED} \
        --output-dir "${SEED_DIR}" \
        2> "${SEED_DIR}/training.log"
    
    echo "Completed seed ${SEED}"
    echo ""
done

echo "============================================================================"
echo "All seeds completed!"
echo "============================================================================"
echo ""

# Generate aggregated plots
echo "Generating aggregated plots with confidence intervals..."
python plot_multiple_seeds.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    --seeds "${SEEDS[@]}"

echo ""
echo "============================================================================"
echo "Experiment completed!"
echo "============================================================================"
echo "Experiment ID: ${EXP_ID}"
echo "Results directory: ${RESULTS_DIR}"
echo "Aggregated plots saved to: ${RESULTS_DIR}/training_curves_with_std.png"
echo "============================================================================"
