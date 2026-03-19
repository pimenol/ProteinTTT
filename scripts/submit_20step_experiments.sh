#!/bin/bash
# Submit 20-step hyperparameter experiments to SLURM (Round 2)
# Current best: neighbors → 0.8222 | Target (100-step best): 0.8416 | Gap: 0.019

set -e

echo "Submitting 20-step experiments (round 2)..."
echo "============================================="

# Exp 8: neighbors + ags=32 (combine the two best individual findings)
echo "[Exp 8] neighbors + ags=32"
sbatch scripts/run_benchmark.sh --config scripts/config_benchmark_exp8_neighbors_ags32.yaml

# Exp 9: neighbors + higher LoRA capacity (alpha=256, rank=128)
echo "[Exp 9] neighbors + alpha=256, rank=128"
sbatch scripts/run_benchmark.sh --config scripts/config_benchmark_exp9_neighbors_a256_r128.yaml

# Exp 10: AdamW with corrected lower lr (0.001 instead of 0.01)
echo "[Exp 10] AdamW lr=0.001"
sbatch scripts/run_benchmark.sh --config scripts/config_benchmark_exp10_adamw_lowlr.yaml

# Exp 11: neighbors + ags=24 (intermediate ags, less time than ags=32)
echo "[Exp 11] neighbors + ags=24"
sbatch scripts/run_benchmark.sh --config scripts/config_benchmark_exp11_neighbors_ags24.yaml

echo ""
echo "All experiments submitted! Monitor with: squeue -u \$USER"
echo "Results will appear in data/benchmark/benchmark_20/"
