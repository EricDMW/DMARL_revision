#!/bin/bash

# ============================================================================
# Multi-Seed Training Script
# ============================================================================
# This script runs training with multiple random seeds and aggregates results
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
MAX_EPISODES=50
DDL=2
N_OBS_NEIGHBORS=1
N_REWARD_NEIGHBORS=1

# Seeds to run (adjust as needed)
# Default: 10 seeds for statistical significance
SEEDS=(42 123 456 789)

# You can customize seeds by modifying the array above
# Example for 5 seeds: SEEDS=(42 123 456 789 1000)

# Number of seeds
N_SEEDS=${#SEEDS[@]}

echo "============================================================================"
echo "Multi-Seed Training Experiment"
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
    
    python main.py \
        --mode train \
        --grid_x ${GRID_X} \
        --grid_y ${GRID_Y} \
        --n_obs_neighbors ${N_OBS_NEIGHBORS} \
        --n_reward_neighbors ${N_REWARD_NEIGHBORS} \
        --ddl ${DDL} \
        --max_episodes ${MAX_EPISODES} \
        --seed ${SEED} \
        --save_dir "${SEED_DIR}" \
        > "${SEED_DIR}/training.log" 2>&1
    
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
