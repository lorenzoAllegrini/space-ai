"""Training utilities for ecoGrow prompt tuning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ...data.plant_datamodule import PlantDataModule
from .prompt_learner import PromptLearnerOpenCLIP


@dataclass
class PromptTuningResult:
    """Container for the training statistics of a single plant."""

    plant: str
    best_epoch: int
    best_val_accuracy: float
    test_accuracy: float
    history: List[Dict[str, float]] = field(default_factory=list)


class PromptTuningTrainer:
    """Simple prompt tuning loop that freezes the CLIP backbone."""

    def __init__(
        self,
        clip_model: nn.Module,
        preprocess_train: Callable,
        preprocess_val: Callable,
        device: Optional[torch.device | str] = None,
        epochs: int = 20,
        lr: float = 5e-3,
        weight_decay: float = 0.0,
        log_every: int = 10,
    ) -> None:
        self.clip_model = clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val
        self.device = torch.device(device) if device is not None else next(clip_model.parameters()).device
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_every = log_every

    def fit(
        self,
        data_module: PlantDataModule,
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        template: str = "a photo of {description}",
    ) -> List[PromptTuningResult]:
        """Train a prompt learner for every plant in ``data_module``."""

        data_module.setup(self.preprocess_train, self.preprocess_val)
        results: List[PromptTuningResult] = []

        for plant in data_module.plants():
            class_prompts = data_module.class_prompts(plant)
            learner = PromptLearnerOpenCLIP(
                self.clip_model,
                class_prompts=class_prompts,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                template=template,
                device=self.device,
            ).to(self.device)

            optimizer = torch.optim.AdamW(learner.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            dataloaders = data_module.dataloaders(plant)

            history: List[Dict[str, float]] = []
            best_val_acc = 0.0
            best_epoch = 0
            best_state = None

            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_one_epoch(learner, optimizer, dataloaders["train"], epoch)
                val_acc = self._evaluate(learner, dataloaders.get("val"))

                history.append({"epoch": float(epoch), "train_loss": train_loss, "val_accuracy": val_acc})

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu() for k, v in learner.state_dict().items()}

            if best_state is not None:
                learner.load_state_dict(best_state)

            test_acc = self._evaluate(learner, dataloaders.get("test"))
            results.append(
                PromptTuningResult(
                    plant=plant,
                    best_epoch=best_epoch,
                    best_val_accuracy=best_val_acc,
                    test_accuracy=test_acc,
                    history=history,
                )
            )

        return results

    def _train_one_epoch(self, learner: PromptLearnerOpenCLIP, optimizer, dataloader, epoch: int) -> float:
        learner.train()
        total_loss = 0.0
        total_samples = 0

        for step, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(images)

            logits = learner(image_features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if step % self.log_every == 0:
                current_loss = total_loss / total_samples if total_samples else 0.0
                print(f"[Epoch {epoch:03d}] step {step:04d} - loss: {current_loss:.4f}")

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _evaluate(self, learner: PromptLearnerOpenCLIP, dataloader) -> float:
        if dataloader is None:
            return 0.0

        learner.eval()
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            image_features = self.clip_model.encode_image(images)
            logits = learner(image_features)
            predictions = logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return correct / total if total else 0.0

