"""PyTorch view of a HuggingFace split, used by the reference trainer and class confusability."""

from datasets import Dataset


class IndexedImageDataset:
	"""Wraps a HuggingFace split as a PyTorch dataset. Returns (image, label, _orig_idx)."""

	def __init__(self, hf_dataset: Dataset, image_col: str, label_col: str, transform):
		self.dataset = hf_dataset
		self.image_col = image_col
		self.label_col = label_col
		self.transform = transform

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, idx: int):
		import torch
		item = self.dataset[idx]
		image = item[self.image_col]
		if image.mode != "RGB":
			image = image.convert("RGB")
		label = int(item[self.label_col])
		orig_idx = int(item["_orig_idx"])
		return self.transform(image), torch.tensor(label, dtype=torch.long), orig_idx
