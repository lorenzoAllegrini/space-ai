"""Command line entry-point for ecoGrow prompt tuning experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch

from spaceai.data import PlantDataModule, PlantDataModuleConfig
from spaceai.models.prompt_learning import PromptTuningTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Root directory containing the plant datasets")
    parser.add_argument("prompts", type=Path, help="JSON file with plant prompts")
    parser.add_argument("--plants", nargs="*", default=None, help="Subset of plants to train on")
    parser.add_argument("--model", default="ViT-B-16", help="OpenCLIP model architecture")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k", help="OpenCLIP pre-trained weights")
    parser.add_argument("--device", default=None, help="Training device (defaults to auto-detect)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers", help="DataLoader workers")
    parser.add_argument("--n-ctx", type=int, default=16, dest="n_ctx", help="Number of context tokens")
    parser.add_argument("--ctx-init", default=None, dest="ctx_init", help="Initial context string")
    parser.add_argument(
        "--template",
        default="a close-up photo of {description}",
        help="Prompt template used to build textual descriptions",
    )
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for the prompt learner")
    parser.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay", help="Optimizer weight decay")
    parser.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios when a plant dataset is not already split",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )

    config = PlantDataModuleConfig(
        data_root=args.data_root,
        prompts_path=args.prompts,
        plant_names=args.plants,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=tuple(args.split_ratio),
    )
    datamodule = PlantDataModule(config)

    trainer = PromptTuningTrainer(
        clip_model=model.to(device),
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    results = trainer.fit(
        datamodule,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        template=args.template,
    )

    for result in results:
        print(
            f"Plant: {result.plant} | best epoch: {result.best_epoch} | "
            f"val acc: {result.best_val_accuracy:.3f} | test acc: {result.test_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()

