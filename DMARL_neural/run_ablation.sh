#!/bin/bash

# ============================================================================
# Ablation Study Script for DMARL
# ============================================================================
# This script runs ablation studies on observation and reward neighborhoods
#
# Ablation 1: Vary observation neighbors (κ_o) while keeping reward neighbors (κ_r) constant
# Ablation 2: Vary reward neighbors (κ_r) while keeping observation neighbors (κ_o) constant
# ============================================================================

set -e  # Exit on error

# Get script directory (DMARL_neural)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to script directory to ensure relative paths work correctly
cd "${SCRIPT_DIR}"

# Base directory for results under DMARL_neural
BASE_DIR="${SCRIPT_DIR}/results"
mkdir -p "${BASE_DIR}"

# Generate running ID (timestamp-based)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_${TIMESTAMP}"
RESULTS_DIR="${BASE_DIR}/${RUN_ID}"
mkdir -p "${RESULTS_DIR}"

# Default parameters
GRID_X=5
GRID_Y=5
MAX_EPISODES=50
DDL=2
CURRENT_REWARD_NEIGHBORS=1  # Current/default reward neighbors value

echo "============================================================================"
echo "DMARL Ablation Study"
echo "============================================================================"
echo "Running ID: ${RUN_ID}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo "============================================================================"
echo ""

# ============================================================================
# Ablation 1: Vary observation neighbors (κ_o) while keeping reward neighbors (κ_r) constant
# ============================================================================
echo "============================================================================"
echo "ABLATION 1: Varying Observation Neighbors (κ_o)"
echo "Keeping reward neighbors (κ_r) = ${CURRENT_REWARD_NEIGHBORS}"
echo "============================================================================"

ABLATION1_DIR="${RESULTS_DIR}/ablation1_obs_neighbors"
mkdir -p "${ABLATION1_DIR}"

for OBS_NEIGHBORS in 1 2 3; do
    echo ""
    echo "----------------------------------------"
    echo "Running: κ_o=${OBS_NEIGHBORS}, κ_r=${CURRENT_REWARD_NEIGHBORS}"
    echo "----------------------------------------"
    
    EXP_DIR="${ABLATION1_DIR}/kappa_o_${OBS_NEIGHBORS}_kappa_r_${CURRENT_REWARD_NEIGHBORS}"
    mkdir -p "${EXP_DIR}"
    
    python main.py \
        --mode train \
        --grid_x ${GRID_X} \
        --grid_y ${GRID_Y} \
        --n_obs_neighbors ${OBS_NEIGHBORS} \
        --n_reward_neighbors ${CURRENT_REWARD_NEIGHBORS} \
        --ddl ${DDL} \
        --max_episodes ${MAX_EPISODES} \
        --save_dir "${EXP_DIR}" \
        > "${EXP_DIR}/training.log" 2>&1
    
    echo "Completed: κ_o=${OBS_NEIGHBORS}, κ_r=${CURRENT_REWARD_NEIGHBORS}"
    echo "Results saved to: ${EXP_DIR}"
done

echo ""
echo "Ablation 1 completed!"
echo ""

# ============================================================================
# Ablation 2: Vary reward neighbors (κ_r) while keeping observation neighbors (κ_o) constant
# ============================================================================
echo "============================================================================"
echo "ABLATION 2: Varying Reward Neighbors (κ_r)"
echo "Keeping observation neighbors (κ_o) = 2"
echo "============================================================================"

ABLATION2_DIR="${RESULTS_DIR}/ablation2_reward_neighbors"
mkdir -p "${ABLATION2_DIR}"

FIXED_OBS_NEIGHBORS=2

# Remove duplicates and sort
REWARD_VALUES=$(echo "1 2 ${CURRENT_REWARD_NEIGHBORS}" | tr ' ' '\n' | sort -u | tr '\n' ' ')

for REWARD_NEIGHBORS in ${REWARD_VALUES}; do
    
    echo ""
    echo "----------------------------------------"
    echo "Running: κ_o=${FIXED_OBS_NEIGHBORS}, κ_r=${REWARD_NEIGHBORS}"
    echo "----------------------------------------"
    
    EXP_DIR="${ABLATION2_DIR}/kappa_o_${FIXED_OBS_NEIGHBORS}_kappa_r_${REWARD_NEIGHBORS}"
    mkdir -p "${EXP_DIR}"
    
    python main.py \
        --mode train \
        --grid_x ${GRID_X} \
        --grid_y ${GRID_Y} \
        --n_obs_neighbors ${FIXED_OBS_NEIGHBORS} \
        --n_reward_neighbors ${REWARD_NEIGHBORS} \
        --ddl ${DDL} \
        --max_episodes ${MAX_EPISODES} \
        --save_dir "${EXP_DIR}" \
        > "${EXP_DIR}/training.log" 2>&1
    
    echo "Completed: κ_o=${FIXED_OBS_NEIGHBORS}, κ_r=${REWARD_NEIGHBORS}"
    echo "Results saved to: ${EXP_DIR}"
done

echo ""
echo "Ablation 2 completed!"
echo ""

# ============================================================================
# Generate comparison plots
# ============================================================================
echo "============================================================================"
echo "Generating comparison plots..."
echo "============================================================================"

python plot_ablation_results.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${RESULTS_DIR}"

# Create a summary file
SUMMARY_FILE="${RESULTS_DIR}/run_summary.txt"
cat > "${SUMMARY_FILE}" << EOF
============================================================================
DMARL Ablation Study - Run Summary
============================================================================
Running ID: ${RUN_ID}
Timestamp: ${TIMESTAMP}
Results Directory: ${RESULTS_DIR}

Experiment Parameters:
  - Grid Size: ${GRID_X} x ${GRID_Y}
  - Max Episodes: ${MAX_EPISODES}
  - Deadline Horizon (ddl): ${DDL}

Ablation 1: Varying Observation Neighbors (κ_o)
  - Fixed κ_r = ${CURRENT_REWARD_NEIGHBORS}
  - Varied κ_o: 1, 2, 3

Ablation 2: Varying Reward Neighbors (κ_r)
  - Fixed κ_o = 2
  - Varied κ_r: 1, 2

Saved Files:
  - Individual training curves: ablation1_obs_neighbors/*/training_curves.png
  - Comparison plots:
    * ablation1_obs_neighbors_comparison.png
    * ablation2_reward_neighbors_comparison.png
  - Training results (JSON): */training_results.json
  - Training logs: */training.log
  - Model checkpoints: */checkpoints/

============================================================================
EOF

echo ""
echo "============================================================================"
echo "Ablation study completed!"
echo "============================================================================"
echo "Running ID: ${RUN_ID}"
echo "Results directory: ${RESULTS_DIR}"
echo ""
echo "Summary of saved files:"
echo "  - Individual training curves: ${ABLATION1_DIR}/*/training_curves.png"
echo "  - Comparison plots:"
echo "    * ${RESULTS_DIR}/ablation1_obs_neighbors_comparison.png"
echo "    * ${RESULTS_DIR}/ablation2_reward_neighbors_comparison.png"
echo "  - Training results (JSON): ${ABLATION1_DIR}/*/training_results.json"
echo "  - Training logs: ${ABLATION1_DIR}/*/training.log"
echo "  - Run summary: ${SUMMARY_FILE}"
echo "============================================================================"

