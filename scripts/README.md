# ProteinTTT Experiments

## Configuration File

The `config.yaml` file contains all experiment settings:

```yaml
# Data settings
df_path: ""                    # Path to your data directory
max_sequence_length: 500
use_true_pdb: false           # Enable if you have ground truth PDBs

# TTT settings
use_msa: false                # Enable MSA (requires .a3m files)
use_gradient_clip: false
steps: 30
learning_rate: 0.04
lora_rank: 64
lora_alpha: 128.0

# Device
device: "cuda"                # or "cpu"
```

## Data Directory Structure

Your data directory should contain:

```
your_data/
├── summary.csv              # Required: columns: id, chain_id, sequence
├── msa/                     # Optional: for MSA mode
│   └── {id}_{chain_id}.a3m
└── pdb/                     # Optional: for validation
    └── {id}_{chain_id}.pdb
```

## Output Structure

Results are saved in your data directory:

```
your_data/
├── predicted_structures/
│   ├── ESMFold_ProteinTTT/   # TTT predictions
│   └── ESMFold/               # Baseline predictions
├── logs/                      # Per-sequence logs
├── plots/                     # Visualization plots
├── results.tsv                # Final results
└── execution.log              # Execution log
```