"""
Summarises the encoder sensitivity experiment: how much each embedding-based metric changes
when the encoder changes, and whether each encoder's metrics still track model performance.

Usage (from repo root, after run.py):
	uv run python -m experiments.pipelines.encoder_sensitivity.analysis

Writes CSV tables and report.md to <output_root>/encoder_sensitivity/analysis and
figures to encoder_sensitivity.analysis.figure_folder. All settings come from experiments/slurm/encoder_sensitivity/config.yaml.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.common import load_experiment_config, resolve_repo_path

# (column, label, higher_is_better)
METRICS = [
	("silhouette", "Silhouette score", True),
	("davies_bouldin", "Davies-Bouldin index", False),
	("intra_class_diversity", "Intra-class diversity", True),
	("label_noise_rate", "Label noise rate", False),
	("near_duplicate_rate_0.98", "Near duplicates (0.98)", False),
	("near_duplicate_rate_0.95", "Near duplicates (0.95)", False),
	("near_duplicate_rate_0.99", "Near duplicates (0.99)", False),
	("near_leakage_share_0.98", "Near-duplicate leakage (0.98)", False),
	("separability_subscore", "Separability sub-score", True),
	("diversity_subscore", "Diversity sub-score", True),
	("annotation_subscore", "Annotation sub-score", True),
]


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Analyse the encoder sensitivity results")
	parser.add_argument("--config", default=None, help="Path to this experiment's config.yaml (defaults to experiments/slurm/encoder_sensitivity/config.yaml)")
	return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
	return min(max(value, low), high)


def spearman(first: pd.Series, second: pd.Series) -> float:
	paired = pd.concat([first, second], axis=1).dropna()
	if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
		return float("nan")
	return float(paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank()))


def bootstrap_interval(first: pd.Series, second: pd.Series, random_generator, repeats: int) -> tuple:
	paired = pd.concat([first, second], axis=1).dropna().to_numpy()
	if len(paired) < 3:
		return float("nan"), float("nan")
	values = []
	for _ in range(repeats):
		sample = paired[random_generator.integers(0, len(paired), len(paired))]
		if len(np.unique(sample[:, 0])) > 1 and len(np.unique(sample[:, 1])) > 1:
			values.append(spearman(pd.Series(sample[:, 0]), pd.Series(sample[:, 1])))
	if not values:
		return float("nan"), float("nan")
	low, high = np.percentile(values, [2.5, 97.5])
	return float(low), float(high)


def load_runs(comparison_root: Path, scoring: Dict[str, float]) -> tuple:
	"""One row per (dataset, encoder) with every metric, plus the flagged label noise images."""
	rows, flagged_images = [], {}
	for metrics_path in sorted(comparison_root.glob("*/*/metrics.json")):
		with open(metrics_path) as metrics_file:
			result = json.load(metrics_file)
		metrics = result["metrics"]
		dataset = result["dataset"].split("/")[-1]
		encoder = result["encoder"]
		separability = metrics.get("feature_separability", {})
		diversity = metrics.get("intra_class_diversity", {})
		noise = metrics.get("label_noise", {})
		row = {
			"dataset": dataset,
			"encoder": encoder,
			"embedding_dimension": result.get("embedding_dimension"),
			"number_of_images": result.get("number_of_images"),
			"silhouette": separability.get("silhouette_score"),
			"davies_bouldin": separability.get("davies_bouldin_index"),
			"intra_class_diversity": diversity.get("mean_diversity"),
			"label_noise_rate": noise.get("estimated_noise_rate"),
		}
		for threshold, near_duplicate in metrics.get("near_duplicate", {}).items():
			row[f"near_duplicate_rate_{threshold}"] = near_duplicate["near_duplicate_rate"]
			row[f"near_leakage_share_{threshold}"] = near_duplicate["cross_split_near_duplicates"] / max(near_duplicate["total_images"], 1)
		# Same formulas as SCORING_FORMULAS.md
		if row["silhouette"] is not None and row["davies_bouldin"] is not None:
			silhouette_part = clamp(row["silhouette"] * scoring["silhouette_scale"], 0, 10)
			davies_bouldin_part = clamp(10 - scoring["davies_bouldin_slope"] * row["davies_bouldin"], 0, 10)
			row["separability_subscore"] = (silhouette_part + davies_bouldin_part) / 2
		if row["intra_class_diversity"] is not None:
			row["diversity_subscore"] = clamp(row["intra_class_diversity"] / scoring["diversity_full_score"] * 10, 0, 10)
		if row["label_noise_rate"] is not None:
			row["annotation_subscore"] = clamp(10 * clamp(1 - row["label_noise_rate"] / scoring["noise_zero_score"], 0, 1) ** scoring["noise_exponent"], 0, 10)
		if result.get("benchmark_row_check"):
			row["benchmark_row_check_mean_cosine"] = result["benchmark_row_check"]["mean_cosine"]
		rows.append(row)
		if "flagged_orig_indices" in noise:
			flagged_images[(dataset, encoder)] = set(noise["flagged_orig_indices"])
	return pd.DataFrame(rows), flagged_images


def load_performance(results_root: Path) -> pd.DataFrame:
	"""Fine-tuned macro F1 (mean of the models) and zero-shot Cohen's kappa (mean of the models) per dataset."""
	performance = pd.DataFrame()
	fine_tuning_summary = results_root / "fine_tuning" / "summary.csv"
	if fine_tuning_summary.exists():
		fine_tuning = pd.read_csv(fine_tuning_summary)
		fine_tuning["dataset"] = fine_tuning.dataset.str.split("/").str[-1]
		performance["fine_tuned_f1"] = fine_tuning.groupby("dataset").macro_f1.mean()

	kappa_rows = []
	for predictions_path in sorted((results_root / "zero_shot").glob("*/*/predictions.json")):
		with open(predictions_path) as predictions_file:
			records = json.load(predictions_file)
		true_labels = np.array([record["true_label"] for record in records])
		predicted_labels = np.array([record["predicted_label"] for record in records])
		observed = float((true_labels == predicted_labels).mean())
		classes = np.unique(true_labels)
		expected = float(sum((true_labels == label).mean() * (predicted_labels == label).mean() for label in classes))
		kappa_rows.append({"dataset": predictions_path.parts[-3], "kappa": (observed - expected) / (1 - expected) if expected < 1 else 0.0})
	if kappa_rows:
		performance["zero_shot_kappa"] = pd.DataFrame(kappa_rows).groupby("dataset").kappa.mean()
	return performance


def rank_agreement(values: pd.DataFrame, encoders: List[str], baseline_encoder: str, analysis_settings: Dict) -> tuple:
	"""Pairwise Spearman between encoders across datasets, for every metric."""
	pair_rows, summary_rows = [], []
	for column, label, _ in METRICS:
		if column not in values:
			continue
		wide = values.pivot(index="dataset", columns="encoder", values=column)
		pair_values = []
		for first, second in itertools.combinations([encoder for encoder in encoders if encoder in wide], 2):
			rho = spearman(wide[first], wide[second])
			pair_rows.append({"metric": column, "encoder_a": first, "encoder_b": second, "spearman": rho,
				"datasets": int(wide[[first, second]].dropna().shape[0])})
			pair_values.append(rho)
		versus_baseline = [
			spearman(wide[baseline_encoder], wide[encoder])
			for encoder in wide.columns if encoder != baseline_encoder and baseline_encoder in wide
		]
		pair_values = [value for value in pair_values if not np.isnan(value)]
		versus_baseline = [value for value in versus_baseline if not np.isnan(value)]
		mean_rho = float(np.mean(pair_values)) if pair_values else float("nan")
		summary_rows.append({
			"metric": column,
			"label": label,
			"mean_pairwise_spearman": mean_rho,
			"min_pairwise_spearman": float(np.min(pair_values)) if pair_values else float("nan"),
			f"mean_spearman_vs_{baseline_encoder}": float(np.mean(versus_baseline)) if versus_baseline else float("nan"),
			"verdict": verdict(mean_rho, analysis_settings),
		})
	return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def verdict(mean_rho: float, analysis_settings: Dict) -> str:
	if np.isnan(mean_rho):
		return "not enough data"
	if mean_rho >= analysis_settings["consistent_spearman"]:
		return "consistent across encoders"
	if mean_rho >= analysis_settings["partly_consistent_spearman"]:
		return "partly consistent"
	return "depends on the encoder"


def value_ranges(values: pd.DataFrame) -> pd.DataFrame:
	"""Typical value of each metric per encoder, to show shifts in scale (which matter for fixed cut-offs)."""
	rows = []
	for (encoder, ), group in values.groupby(["encoder"]):
		for column, label, _ in METRICS:
			if column in group and group[column].notna().any():
				rows.append({"encoder": encoder, "metric": column, "median": group[column].median(),
					"p10": group[column].quantile(0.1), "p90": group[column].quantile(0.9), "datasets": int(group[column].notna().sum())})
	return pd.DataFrame(rows)


def flagged_overlap(flagged_images: Dict[tuple, set], encoders: List[str]) -> pd.DataFrame:
	"""Mean Jaccard overlap of the images flagged as likely mislabeled, per encoder pair."""
	datasets = sorted({dataset for dataset, _ in flagged_images})
	rows = []
	for first, second in itertools.combinations(encoders, 2):
		overlaps = []
		for dataset in datasets:
			first_set, second_set = flagged_images.get((dataset, first)), flagged_images.get((dataset, second))
			if first_set is None or second_set is None or not (first_set | second_set):
				continue
			overlaps.append(len(first_set & second_set) / len(first_set | second_set))
		if overlaps:
			rows.append({"encoder_a": first, "encoder_b": second, "mean_jaccard": float(np.mean(overlaps)), "datasets": len(overlaps)})
	return pd.DataFrame(rows)


def predictive_power(values: pd.DataFrame, performance: pd.DataFrame, encoders: List[str], analysis_settings: Dict) -> pd.DataFrame:
	random_generator = np.random.default_rng(analysis_settings["bootstrap_seed"])
	rows = []
	for target in [column for column in ["fine_tuned_f1", "zero_shot_kappa"] if column in performance]:
		for encoder in encoders:
			encoder_values = values[values.encoder == encoder].set_index("dataset")
			for column, label, _ in METRICS:
				if column not in encoder_values:
					continue
				joined = encoder_values[[column]].join(performance[[target]], how="inner").dropna()
				low, high = bootstrap_interval(joined[column], joined[target], random_generator, analysis_settings["bootstrap_repeats"])
				rows.append({"target": target, "encoder": encoder, "metric": column, "spearman": spearman(joined[column], joined[target]),
					"interval_low": low, "interval_high": high, "datasets": len(joined)})
	return pd.DataFrame(rows)


def draw_figures(pairs: pd.DataFrame, predictive: pd.DataFrame, encoders: List[str], figure_folder: Path, heatmap_metrics: List[str]) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	figure_folder.mkdir(parents=True, exist_ok=True)
	labels = {column: label for column, label, _ in METRICS}

	# Pairwise rank agreement between encoders, one panel per metric
	panels = [metric for metric in heatmap_metrics if metric in set(pairs.metric)]
	if panels:
		figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
		axes = np.atleast_1d(axes)
		for axis, metric in zip(axes, panels):
			matrix = pd.DataFrame(np.eye(len(encoders)), index=encoders, columns=encoders)
			for _, pair in pairs[pairs.metric == metric].iterrows():
				matrix.loc[pair.encoder_a, pair.encoder_b] = matrix.loc[pair.encoder_b, pair.encoder_a] = pair.spearman
			image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu")
			axis.set_xticks(range(len(encoders)), encoders, rotation=60, ha="right", fontsize=8)
			axis.set_yticks(range(len(encoders)), encoders if axis is axes[0] else [""] * len(encoders), fontsize=8)
			for row_index, column_index in itertools.product(range(len(encoders)), repeat=2):
				value = matrix.iloc[row_index, column_index]
				if not np.isnan(value):
					axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
						color="white" if abs(value) > 0.6 else "black")
			axis.set_title(labels[metric], fontsize=10)
		figure.colorbar(image, ax=axes, shrink=0.7, label="Spearman rho across datasets")
		figure.savefig(figure_folder / "rank_agreement_between_encoders.png", dpi=200, bbox_inches="tight")
		plt.close(figure)

	# Correlation with model performance per encoder
	for target, title in [("fine_tuned_f1", "fine-tuned macro F1"), ("zero_shot_kappa", "zero-shot Cohen's kappa")]:
		subset = predictive[(predictive.target == target) & predictive.metric.isin(heatmap_metrics + ["separability_subscore", "annotation_subscore"])]
		if subset.empty:
			continue
		matrix = subset.pivot(index="metric", columns="encoder", values="spearman").reindex(columns=[encoder for encoder in encoders if encoder in set(subset.encoder)])
		figure, axis = plt.subplots(figsize=(1.1 * matrix.shape[1] + 3, 0.55 * matrix.shape[0] + 1.5))
		axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu")
		axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right", fontsize=8)
		axis.set_yticks(range(matrix.shape[0]), [labels[metric] for metric in matrix.index], fontsize=8)
		for row_index, column_index in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
			value = matrix.iloc[row_index, column_index]
			if not np.isnan(value):
				axis.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=7,
					color="white" if abs(value) > 0.6 else "black")
		axis.set_title(f"Spearman rho with {title}, per encoder", fontsize=10)
		figure.savefig(figure_folder / f"correlation_with_{target}.png", dpi=200, bbox_inches="tight")
		plt.close(figure)


def format_table(frame: pd.DataFrame, columns: List[str]) -> str:
	header = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
	lines = []
	for _, row in frame[columns].iterrows():
		cells = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in row]
		lines.append("| " + " | ".join(cells) + " |")
	return header + "\n".join(lines) + "\n"


def write_report(summary: pd.DataFrame, overlap: pd.DataFrame, predictive: pd.DataFrame, values: pd.DataFrame,
		encoders: List[str], analysis_folder: Path, baseline_encoder: str, analysis_settings: Dict) -> None:
	lines = ["# Encoder sensitivity", "",
		f"Encoders: {', '.join(encoders)}. Datasets: {values.dataset.nunique()}.", "",
		"## Do encoders rank the datasets the same way?", "",
		f"Mean and lowest pairwise Spearman rho between encoders, across datasets. {analysis_settings['consistent_spearman']} or more counts as consistent.", "",
		format_table(summary, ["label", "mean_pairwise_spearman", "min_pairwise_spearman", f"mean_spearman_vs_{baseline_encoder}", "verdict"])]
	if not overlap.empty:
		lines += ["## Do encoders flag the same images as mislabeled?", "",
			f"Mean Jaccard overlap of flagged images over all encoder pairs: {overlap.mean_jaccard.mean():.2f} "
			f"(lowest pair {overlap.mean_jaccard.min():.2f}, highest {overlap.mean_jaccard.max():.2f}).", ""]
	if "benchmark_row_check_mean_cosine" in values:
		check = values.benchmark_row_check_mean_cosine.dropna()
		if not check.empty:
			lines += ["## Row check against the benchmark", "",
				f"{baseline_encoder} embeddings vs the saved benchmark embeddings: mean cosine {check.mean():.4f}, lowest dataset {check.min():.4f}.", ""]
	for target, title in [("fine_tuned_f1", "fine-tuned macro F1"), ("zero_shot_kappa", "zero-shot kappa")]:
		subset = predictive[(predictive.target == target)]
		if subset.empty:
			continue
		wide = subset.pivot(index="metric", columns="encoder", values="spearman").reindex(columns=[e for e in encoders if e in set(subset.encoder)])
		wide.insert(0, "metric", wide.index)
		lines += [f"## Correlation with {title}", "", f"Spearman rho on the datasets with model results (n = {int(subset.datasets.max())}).", "",
			format_table(wide.reset_index(drop=True), list(wide.columns))]
	(analysis_folder / "report.md").write_text("\n".join(lines))


def main() -> None:
	arguments = parse_arguments()
	experiment_config = load_experiment_config("encoder_sensitivity", arguments.config)
	results_root = Path(experiment_config["shared"]["output_root"])
	comparison_root = results_root / experiment_config["encoder_sensitivity"]["experiment_name"]
	analysis_folder = comparison_root / "analysis"
	analysis_folder.mkdir(parents=True, exist_ok=True)
	settings = experiment_config["encoder_sensitivity"]
	analysis_settings = settings["analysis"]
	baseline_encoder = settings["baseline_encoder"]
	configured_encoders = [encoder["name"] for encoder in settings["encoders"]]

	values, flagged_images = load_runs(comparison_root, settings["scoring"])
	if values.empty:
		raise SystemExit(f"No metrics.json files found under {comparison_root}")
	encoders = [encoder for encoder in configured_encoders if encoder in set(values.encoder)]
	performance = load_performance(results_root)

	pairs, summary = rank_agreement(values, encoders, baseline_encoder, analysis_settings)
	ranges = value_ranges(values)
	overlap = flagged_overlap(flagged_images, encoders)
	predictive = predictive_power(values, performance, encoders, analysis_settings) if not performance.empty else pd.DataFrame()

	values.to_csv(analysis_folder / "metric_values.csv", index=False)
	pairs.to_csv(analysis_folder / "rank_agreement_pairs.csv", index=False)
	summary.to_csv(analysis_folder / "rank_agreement_summary.csv", index=False)
	ranges.to_csv(analysis_folder / "value_ranges.csv", index=False)
	overlap.to_csv(analysis_folder / "label_noise_flag_overlap.csv", index=False)
	predictive.to_csv(analysis_folder / "correlation_with_performance.csv", index=False)
	write_report(summary, overlap, predictive, values, encoders, analysis_folder, baseline_encoder, analysis_settings)
	figure_folder = resolve_repo_path(analysis_settings["figure_folder"])
	draw_figures(pairs, predictive, encoders, figure_folder, analysis_settings["heatmap_metrics"])

	print(summary[["label", "mean_pairwise_spearman", "min_pairwise_spearman", "verdict"]].round(2).to_string(index=False))
	print(f"\nTables and report.md: {analysis_folder}")


if __name__ == "__main__":
	main()
