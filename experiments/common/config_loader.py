"""Reads experiments/slurm/shared.yaml and merges in one experiment's experiments/slurm/<experiment>/config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_FOLDER = REPO_ROOT / "experiments" / "slurm"
SHARED_CONFIG_PATH = SLURM_FOLDER / "shared.yaml"


@dataclass
class DatasetEntry:
	name: str
	label_column: Optional[str] = None
	image_column: Optional[str] = None
	hf_config_name: Optional[str] = None
	compound_label_columns: Optional[List[str]] = None
	class_name_overrides: Optional[Dict[str, str]] = None

	@property
	def short_name(self) -> str:
		return self.name.split("/")[-1]


def read_yaml(path: Path) -> Dict[str, Any]:
	with open(path) as yaml_file:
		return yaml.safe_load(yaml_file) or {}


def load_experiment_config(experiment_name: str, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
	"""Shared settings and datasets from shared.yaml, plus the experiment's config.yaml. Top-level keys in the
	experiment's file (its own section, or a datasets list of its own) replace the shared ones."""
	experiment_path = resolve_repo_path(config_path) if config_path else SLURM_FOLDER / experiment_name / "config.yaml"
	experiment_config = {**read_yaml(SHARED_CONFIG_PATH), **read_yaml(experiment_path)}
	if experiment_name not in experiment_config:
		raise KeyError(f"{experiment_path} has no '{experiment_name}' section")
	experiment_config["shared"]["output_root"] = str(resolve_repo_path(experiment_config["shared"]["output_root"]))
	return experiment_config


def resolve_repo_path(raw_path: Union[str, Path]) -> Path:
	"""Expands ~ and $VARS. Relative paths are taken from the repo root, so the working directory never matters."""
	path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
	return path if path.is_absolute() else REPO_ROOT / path


def parse_dataset_entries(
	experiment_config: Dict[str, Any],
	selected_dataset_names: Optional[List[str]] = None,
) -> List[DatasetEntry]:
	"""Merges dataset_defaults into every dataset entry. Selection accepts full or short names."""
	dataset_defaults = experiment_config.get("dataset_defaults") or {}
	dataset_entries = []
	for raw_entry in experiment_config["datasets"]:
		merged_entry = {**dataset_defaults, **raw_entry}
		dataset_entries.append(DatasetEntry(
			name=merged_entry["name"],
			label_column=merged_entry.get("label_column") or None,
			image_column=merged_entry.get("image_column") or None,
			hf_config_name=merged_entry.get("hf_config_name") or None,
			compound_label_columns=merged_entry.get("compound_label_columns") or None,
			class_name_overrides=merged_entry.get("class_name_overrides") or None,
		))

	if not selected_dataset_names:
		return dataset_entries

	wanted_names = set(selected_dataset_names)
	selected_entries = [
		entry for entry in dataset_entries
		if entry.name in wanted_names or entry.short_name in wanted_names
	]
	known_names = {entry.name for entry in dataset_entries} | {entry.short_name for entry in dataset_entries}
	unknown_names = wanted_names - known_names
	if unknown_names:
		raise ValueError(f"Datasets not found in the dataset list: {sorted(unknown_names)}")
	return selected_entries
