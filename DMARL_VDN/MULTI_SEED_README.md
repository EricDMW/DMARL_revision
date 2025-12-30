# Multi-Seed Training and Plotting for VDN

This directory contains scripts for running VDN training with multiple random seeds and generating publication-quality plots with confidence intervals.

## Files

- `run_multiple_seeds.sh`: Shell script that runs training with multiple seeds
- `plot_multiple_seeds.py`: Python script that aggregates results and creates plots
- `MULTI_SEED_README.md`: This file

## Usage

### Running Multiple Seeds

```bash
# Make the script executable (if not already)
chmod +x run_multiple_seeds.sh

# Run training with multiple seeds
./run_multiple_seeds.sh
```

The script will:
1. Create a timestamped results directory under `results/`
2. Run training for each seed sequentially
3. Save results for each seed in separate subdirectories
4. Automatically generate aggregated plots with confidence intervals

### Customizing Seeds

Edit `run_multiple_seeds.sh` to change the seeds:

```bash
# Example: 5 seeds
SEEDS=(42 123 456 789 1000)

# Example: 10 seeds (default)
SEEDS=(42 123 456 789 1000 2000 3000 4000 5000 6000)
```

### Customizing Parameters

Edit `run_multiple_seeds.sh` to change training parameters:

```bash
GRID_X=5
GRID_Y=5
MAX_EPISODES=5000
DDL=2
N_OBS_NEIGHBORS=1
N_REWARD_NEIGHBORS=1
```

### Generating Plots Manually

If you need to regenerate plots from existing results:

```bash
python plot_multiple_seeds.py \
    --results_dir results/seeds_YYYYMMDD_HHMMSS \
    --seeds 42 123 456 789 1000 \
    --window 100 \
    --show_individual
```

Options:
- `--results_dir`: Directory containing seed subdirectories
- `--seeds`: List of seed values used
- `--window`: Moving average window size (default: 100)
- `--show_individual`: Show individual seed curves (faded) in addition to mean/std

## Output Structure

```
results/
└── seeds_YYYYMMDD_HHMMSS/
    ├── seed_42/
    │   ├── training.log
    │   ├── training_results.json
    │   ├── training_curves.png
    │   └── checkpoints/
    ├── seed_123/
    │   └── ...
    ├── ...
    ├── training_curves_with_std.png  (aggregated plot)
    └── aggregated_stats.json         (statistics summary)
```

## Plot Features

The generated plots include:

1. **Mean Curve**: Average performance across all seeds
2. **Confidence Interval**: Shaded region showing ±1 standard deviation
3. **Moving Average**: Optional smoothing for cleaner visualization
4. **Academic Style**: 
   - Professional color scheme
   - High resolution (300 DPI)
   - Clean typography (serif fonts)
   - Publication-ready format

### Plot Style

- **Font**: Times New Roman (serif) for academic papers
- **Resolution**: 300 DPI for high-quality figures
- **Colors**: Professional palette suitable for colorblind readers
- **Grid**: Subtle grid lines for readability
- **Spines**: Clean appearance (no top/right borders)

## Statistical Information

The script computes:
- Mean performance across seeds
- Standard deviation (confidence interval)
- Final performance statistics
- Maximum achieved performance

All statistics are saved in `aggregated_stats.json`.

## Example Output

The plot shows:
- **Solid line**: Mean episode reward across all seeds
- **Shaded region**: ±1 standard deviation (confidence interval)
- **Optional**: Individual seed curves (faded) for transparency

This format is standard in top-tier conferences (NeurIPS, ICML, ICLR, etc.).

## Notes

- Each seed runs sequentially, so total time is the sum of individual training times
- Results are saved with timestamps to avoid overwriting
- The plotting script automatically handles different episode lengths (truncates to minimum)
- All plots are saved in PNG format with high resolution suitable for papers
