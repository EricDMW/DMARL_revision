# Ablation Study for DMARL

This directory contains scripts to run ablation studies on the DMARL algorithm, specifically examining the effects of observation neighborhood (κ_o) and reward neighborhood (κ_r) parameters.

## Files

- `run_ablation.sh`: Shell script that runs all ablation experiments
- `plot_ablation_results.py`: Python script that generates comparison plots from results
- `ABLATION_README.md`: This file

## Ablation Studies

### Ablation 1: Varying Observation Neighbors (κ_o)
- **Fixed parameter**: κ_r = 1 (current/default reward neighbors)
- **Varied parameter**: κ_o ∈ {1, 2, 3}
- **Purpose**: Examine the effect of observation range on learning performance

### Ablation 2: Varying Reward Neighbors (κ_r)
- **Fixed parameter**: κ_o = 2
- **Varied parameter**: κ_r ∈ {1, 2}
- **Purpose**: Examine the effect of reward aggregation range on learning performance

## Usage

### Running the Ablation Study

```bash
# Make sure the script is executable
chmod +x run_ablation.sh

# Run the ablation study
./run_ablation.sh
```

The script will:
1. Create a results directory under `DMARL_neural/results/` with a unique running ID
2. Run all experiments sequentially
3. Save training logs, results (JSON), and figures for each experiment
4. Automatically generate comparison plots at the end
5. All figures (training curves and comparison plots) are saved in the results directory

### Customizing Parameters

You can edit `run_ablation.sh` to customize:
- Grid size (`GRID_X`, `GRID_Y`)
- Number of episodes (`MAX_EPISODES`)
- Deadline horizon (`DDL`)
- Current reward neighbors value (`CURRENT_REWARD_NEIGHBORS`)

### Generating Plots Manually

If you need to regenerate plots from existing results:

```bash
python plot_ablation_results.py \
    --results_dir results/run_YYYYMMDD_HHMMSS \
    --output_dir results/run_YYYYMMDD_HHMMSS
```

## Output Structure

Results are saved under `DMARL_neural/results/` with a unique running ID:

```
DMARL_neural/
└── results/
    └── run_YYYYMMDD_HHMMSS/
        ├── ablation1_obs_neighbors/
        │   ├── kappa_o_1_kappa_r_1/
        │   │   ├── training.log
        │   │   ├── training_results.json
        │   │   ├── training_curves.png
        │   │   └── checkpoints/
        │   ├── kappa_o_2_kappa_r_1/
        │   └── kappa_o_3_kappa_r_1/
        ├── ablation2_reward_neighbors/
        │   ├── kappa_o_2_kappa_r_1/
        │   └── kappa_o_2_kappa_r_2/
        ├── ablation1_obs_neighbors_comparison.png
        └── ablation2_reward_neighbors_comparison.png
```

Each run gets a unique ID based on timestamp (format: `run_YYYYMMDD_HHMMSS`), ensuring no overwrites.

## Results Format

Each experiment saves:
- `training.log`: Full training log with progress
- `training_results.json`: Structured results including:
  - Episode rewards (full history)
  - Episode lengths
  - Critic and actor losses
  - Best reward achieved
  - Configuration parameters
- `training_curves.png`: Individual training curves
- `checkpoints/`: Model checkpoints

## Comparison Plots

The generated comparison plots show:
- **Ablation 1 plot**: Training curves for different κ_o values (fixed κ_r=1)
- **Ablation 2 plot**: Training curves for different κ_r values (fixed κ_o=2)

Both plots use moving averages (100 episodes) for smoother visualization.

## Notes

- Each experiment runs sequentially, so total time is the sum of all individual experiment times
- Results are saved with unique running IDs (timestamps) to avoid overwriting previous runs
- All figures (training curves and comparison plots) are automatically saved to the results directory
- Training logs are redirected to files to keep console output clean
- The plotting script automatically detects and loads results from JSON files
- The `results/` directory is created automatically if it doesn't exist

