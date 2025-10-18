"""PyTorch datamodule tailored for the ecoGrow prompt tuning pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .plant_dataset import PlantDataset


def _default_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    eval_transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    return train_transform, eval_transform


@dataclass
class PlantDataModuleConfig:
    """Configuration parameters for :class:`PlantDataModule`."""

    data_root: Path
    prompts_path: Path
    plant_names: Optional[Iterable[str]] = None
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42
    segmentation_fns: Optional[Mapping[str, Callable]] = None


class PlantDataModule:
    """Build plant-specific dataloaders for prompt tuning experiments."""

    def __init__(self, config: PlantDataModuleConfig) -> None:
        self.config = config
        self.data_root = config.data_root.expanduser().resolve()
        self.prompts_path = config.prompts_path.expanduser().resolve()

        if not self.data_root.exists():
            msg = f"Dataset directory '{self.data_root}' does not exist"
            raise FileNotFoundError(msg)

        with self.prompts_path.open("r", encoding="utf-8") as handle:
            self.prompts_config: Dict[str, Dict[str, str]] = json.load(handle)

        if config.plant_names is None:
            plant_names = sorted(self.prompts_config.keys())
        else:
            plant_names = list(config.plant_names)

        self.plant_names = plant_names
        self._train_transform: Optional[Callable] = None
        self._eval_transform: Optional[Callable] = None
        self._datasets: Dict[str, Dict[str, Dataset]] = {}

    def setup(
        self,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
    ) -> None:
        """Instantiate the datasets for every plant species."""

        if train_transform is None or eval_transform is None:
            train_transform, eval_transform = _default_transforms(self.config.image_size)

        self._train_transform = train_transform
        self._eval_transform = eval_transform

        rng = random.Random(self.config.seed)

        for plant in self.plant_names:
            class_prompts = self.prompts_config.get(plant)
            if not class_prompts:
                msg = f"No prompt entries found for plant '{plant}'"
                raise KeyError(msg)

            class_to_idx = {class_name: idx for idx, class_name in enumerate(class_prompts)}
            plant_dir = self.data_root / plant
            if not plant_dir.exists():
                msg = f"Plant directory '{plant_dir}' does not exist"
                raise FileNotFoundError(msg)

            segmentation_fn = None
            if self.config.segmentation_fns is not None:
                segmentation_fn = self.config.segmentation_fns.get(plant)

            datasets: Dict[str, Dataset] = {}

            split_dirs = {name: plant_dir / name for name in ("train", "val", "test")}
            if all(path.exists() for path in split_dirs.values()):
                datasets["train"] = PlantDataset.from_directory(
                    split_dirs["train"],
                    class_to_idx,
                    transform=train_transform,
                    segmentation_fn=segmentation_fn,
                )
                datasets["val"] = PlantDataset.from_directory(
                    split_dirs["val"],
                    class_to_idx,
                    transform=eval_transform,
                    segmentation_fn=segmentation_fn,
                )
                datasets["test"] = PlantDataset.from_directory(
                    split_dirs["test"],
                    class_to_idx,
                    transform=eval_transform,
                    segmentation_fn=segmentation_fn,
                )
            else:
                base_dataset = PlantDataset.from_directory(
                    plant_dir,
                    class_to_idx,
                    transform=None,
                    segmentation_fn=segmentation_fn,
                )

                lengths = self._compute_split_lengths(len(base_dataset))
                indices = list(range(len(base_dataset)))
                rng.shuffle(indices)

                split_offsets = [0, lengths[0], lengths[0] + lengths[1], len(base_dataset)]
                split_indices = {
                    "train": indices[split_offsets[0] : split_offsets[1]],
                    "val": indices[split_offsets[1] : split_offsets[2]],
                    "test": indices[split_offsets[2] : split_offsets[3]],
                }

                for split_name, idxs in split_indices.items():
                    split_transform = train_transform if split_name == "train" else eval_transform
                    datasets[split_name] = PlantDataset(
                        image_paths=[base_dataset.image_paths[i] for i in idxs],
                        labels=[base_dataset.labels[i] for i in idxs],
                        class_names=base_dataset.class_names,
                        transform=split_transform,
                        segmentation_fn=base_dataset.segmentation_fn,
                    )

            self._datasets[plant] = datasets

    def dataloaders(self, plant: str) -> Dict[str, DataLoader]:
        """Return the train/val/test dataloaders for a given plant."""

        if plant not in self._datasets:
            msg = "Call setup() before requesting dataloaders"
            raise RuntimeError(msg)

        datasets = self._datasets[plant]
        dataloaders: Dict[str, DataLoader] = {}
        for split_name, dataset in datasets.items():
            shuffle = split_name == "train"
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        return dataloaders

    def class_prompts(self, plant: str) -> Dict[str, str]:
        """Return the prompt dictionary associated with ``plant``."""

        prompts = self.prompts_config.get(plant)
        if prompts is None:
            msg = f"Unknown plant '{plant}'"
            raise KeyError(msg)
        return prompts

    def plants(self) -> Iterator[str]:
        """Iterate over the configured plant species."""

        yield from self.plant_names

    def _compute_split_lengths(self, dataset_length: int) -> Tuple[int, int, int]:
        train_ratio, val_ratio, test_ratio = self.config.split_ratio
        total = train_ratio + val_ratio + test_ratio
        if total <= 0:
            msg = "Split ratios must add up to a positive number"
            raise ValueError(msg)

        train_len = int(dataset_length * train_ratio / total)
        val_len = int(dataset_length * val_ratio / total)
        test_len = dataset_length - train_len - val_len
        if train_len == 0 or val_len == 0 or test_len == 0:
            msg = "Dataset too small for the requested split ratios"
            raise ValueError(msg)

        return train_len, val_len, test_len

