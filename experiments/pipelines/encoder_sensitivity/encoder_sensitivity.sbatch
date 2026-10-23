#!/bin/bash -l
#SBATCH --job-name=agml_benchmarking_encoder_sensitivity
#SBATCH --account=jmearlesgrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err

# Submit from the repo root. Extra arguments are passed straight to the runner:
#   sbatch experiments/pipelines/encoder_sensitivity/encoder_sensitivity.sbatch
#   sbatch experiments/pipelines/encoder_sensitivity/encoder_sensitivity.sbatch --encoders dinov2_base,clip_vit_large --datasets bean_disease_uganda
#
# Compute nodes do not have reliable internet, so cache datasets and weights on the login node first:
#   uv run python -m experiments.pipelines.encoder_sensitivity.run --prefetch-only
#   uv run python -m experiments.pipelines.encoder_sensitivity.run --smoke-test    (checks that every encoder loads and encodes)
#
# DINOv3 is gated: accept its license on Hugging Face and run huggingface-cli login before prefetching.
# When the job is done, summarise the results with:
#   uv run python -m experiments.pipelines.encoder_sensitivity.analysis

module load cuda/12

export UV_PROJECT_ENVIRONMENT=/group/jmearlesgrp/$USER/agml-benchmarking/.agml-benchmarking/
export HF_HOME=/group/jmearlesgrp/uchandar/.cache/huggingface
export AGML_DATA_DIR=/group/jmearlesgrp/uchandar/.cache
export TORCH_HOME=/group/jmearlesgrp/uchandar/.cache/torch
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "Working directory : $PWD"
echo "Node              : $(hostname)"
echo "Started           : $(date)"
echo "---------------------------------------------------------"

uv run python -m experiments.pipelines.encoder_sensitivity.run --config "$REPO_ROOT/experiments/config.yaml" "$@"

echo "---------------------------------------------------------"
echo "Finished : $(date)"
