"""Dataset utilities for the ecoGrow plant disease recognition project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _default_loader(path: Path) -> Image.Image:
    """Load an image from *path* and convert it to RGB."""

    with Image.open(path) as img:
        return img.convert("RGB")


@dataclass
class Sample:
    """A single dataset sample returned by :class:`PlantDataset`.

    Attributes
    ----------
    image:
        The pre-processed PIL image.
    label:
        Integer index identifying the target class.
    class_name:
        Human readable name of the target class. This is useful when
        aggregating few-shot results across different plants.
    path:
        Path to the original image on disk.
    """

    image: Image.Image
    label: int
    class_name: str
    path: Path


class PlantDataset(Dataset[Tuple[Image.Image, int]]):
    """A lightweight dataset wrapping plant disease images.

    Parameters
    ----------
    image_paths:
        Sequence of image paths to load.
    labels:
        Sequence with the class index associated to every element in
        ``image_paths``.
    class_names:
        Mapping from integer labels to their class names. The index of the
        list corresponds to the numeric label stored in ``labels``.
    transform:
        Optional callable applied to the PIL image. Usually this comes from
        the OpenCLIP preprocessing utilities.
    segmentation_fn:
        Optional callable applied to the PIL image before the transforms in
        order to isolate the plant from the background.
    loader:
        Callable responsible for loading raw images from disk. Defaults to a
        simple RGB loader based on :mod:`PIL`.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        class_names: Sequence[str],
        transform: Optional[Callable[[Image.Image], object]] = None,
        segmentation_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        loader: Callable[[Path], Image.Image] = _default_loader,
    ) -> None:
        if len(image_paths) != len(labels):
            msg = "image_paths and labels must contain the same number of items"
            raise ValueError(msg)

        self.image_paths: List[Path] = list(image_paths)
        self.labels: List[int] = list(labels)
        self.class_names: List[str] = list(class_names)
        self.transform = transform
        self.segmentation_fn = segmentation_fn
        self.loader = loader

    def __len__(self) -> int:  # noqa: D401 - standard Dataset API
        """Return the number of available samples."""

        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[object, int]:  # noqa: D401
        """Return the processed sample at ``index``.

        The returned tuple contains the transformed image and the
        corresponding class index. The image can either be a PIL image or a
        tensor depending on the transform passed at construction time.
        """

        path = self.image_paths[index]
        label = self.labels[index]

        image = self.loader(path)
        if self.segmentation_fn is not None:
            image = self.segmentation_fn(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_directory(
        cls,
        root: Path,
        class_to_idx: Dict[str, int],
        transform: Optional[Callable[[Image.Image], object]] = None,
        segmentation_fn: Optional[Callable[[Image.Image], Image.Image]] = None,
        extensions: Optional[Iterable[str]] = None,
    ) -> "PlantDataset":
        """Create a dataset by scanning a directory tree.

        The directory must contain one sub-directory per class. Each
        sub-directory is expected to host the images for that specific class.
        The resulting dataset keeps the alphabetical order of classes unless
        ``class_to_idx`` specifies a custom mapping.
        """

        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        else:
            extensions = {ext.lower() for ext in extensions}

        root = root.expanduser().resolve()
        if not root.exists():
            msg = f"Plant directory '{root}' does not exist"
            raise FileNotFoundError(msg)

        image_paths: List[Path] = []
        labels: List[int] = []
        class_names: List[str] = [""] * len(class_to_idx)

        for class_name, class_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            class_dir = root / class_name
            if not class_dir.exists():
                continue

            class_names[class_idx] = class_name
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in extensions and path.is_file():
                    image_paths.append(path)
                    labels.append(class_idx)

        if not image_paths:
            msg = f"No images found inside '{root}'"
            raise RuntimeError(msg)

        return cls(
            image_paths=image_paths,
            labels=labels,
            class_names=class_names,
            transform=transform,
            segmentation_fn=segmentation_fn,
        )

