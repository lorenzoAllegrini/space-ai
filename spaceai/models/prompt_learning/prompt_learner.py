"""Prompt learner module inspired by CoOp for the ecoGrow project."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

import open_clip


class PromptLearnerOpenCLIP(nn.Module):
    """Learn context vectors to adapt OpenCLIP prompts.

    Parameters
    ----------
    clip_model:
        The OpenCLIP model whose text tower is leveraged to encode prompts.
        The caller is expected to keep the backbone in evaluation mode and to
        freeze all parameters before instantiating the learner.
    class_prompts:
        Mapping between class identifiers and their textual descriptions. The
        insertion order is preserved and defines the class index order.
    n_ctx:
        Number of learnable context tokens prepended to each class prompt.
    ctx_init:
        Optional string used to initialise the context vectors using the
        pre-trained CLIP token embeddings. When omitted, the contexts are
        sampled from a normal distribution with a standard deviation of 0.02.
    template:
        Template string where ``{description}`` is replaced with the class
        description. Defaults to "a photo of {description}".
    device:
        Optional device identifier used to host the learnable parameters.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        class_prompts: Dict[str, str],
        n_ctx: int = 16,
        ctx_init: Optional[str] = None,
        template: str = "a photo of {description}",
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        if n_ctx <= 0:
            msg = "n_ctx must be a positive integer"
            raise ValueError(msg)

        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.template = template
        self.device = device or next(clip_model.parameters()).device

        self.class_names: List[str] = list(class_prompts.keys())
        self.class_descriptions: List[str] = list(class_prompts.values())
        if not self.class_names:
            msg = "class_prompts cannot be empty"
            raise ValueError(msg)

        with torch.no_grad():
            token_embedding = getattr(self.clip_model, "token_embedding")
            if token_embedding is None:
                msg = "The provided clip_model does not expose token_embedding"
                raise AttributeError(msg)
            embed_dim = token_embedding.weight.shape[1]
            embedding_dtype = token_embedding.weight.dtype

        initial_context = self._build_initial_context(ctx_init, embed_dim, embedding_dtype)
        self.ctx = nn.Parameter(initial_context)

        prompt_texts = [self._build_prompt_text(description) for description in self.class_descriptions]
        self.register_buffer("tokenized_prompts", open_clip.tokenize(prompt_texts))

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Compute image-text logits for the provided ``image_features``."""

        text_features = self.encode_text()
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    def encode_text(self) -> torch.Tensor:
        """Encode the class prompts with the current context vectors."""

        tokenized = self.tokenized_prompts.to(self.device)
        embeddings = self.clip_model.token_embedding(tokenized).to(self.ctx.dtype)

        ctx = self.ctx.unsqueeze(0).expand(tokenized.size(0), -1, -1)
        embeddings[:, 1 : 1 + self.n_ctx, :] = ctx

        positional_embedding = self.clip_model.positional_embedding.to(embeddings.dtype)
        x = embeddings + positional_embedding
        x = x.permute(1, 0, 2)
        attn_mask = getattr(self.clip_model, "attn_mask", None)
        if attn_mask is not None:
            x = self.clip_model.transformer(x, attn_mask=attn_mask)
        else:
            x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)

        eos_token = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_token]
        text_features = x @ self.clip_model.text_projection

        return text_features

    def _build_initial_context(
        self,
        ctx_init: Optional[str],
        embed_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ctx_init:
            tokenized = open_clip.tokenize([ctx_init]).to(self.device)
            with torch.no_grad():
                embeddings = self.clip_model.token_embedding(tokenized)[0, 1 : 1 + self.n_ctx]
            if embeddings.shape[0] < self.n_ctx:
                msg = "ctx_init is too short for the requested number of context tokens"
                raise ValueError(msg)
            ctx_vectors = embeddings[: self.n_ctx]
        else:
            ctx_vectors = torch.randn(self.n_ctx, embed_dim, device=self.device, dtype=dtype) * 0.02

        return ctx_vectors.to(self.device, dtype=dtype)

    def _build_prompt_text(self, description: str) -> str:
        context_placeholder = " ".join(["X"] * self.n_ctx)
        prompt = self.template.format(description=description)
        return f"{context_placeholder} {prompt}".strip()

    @property
    def class_prompts(self) -> Dict[str, str]:
        """Return the mapping between class names and their descriptions."""

        return dict(zip(self.class_names, self.class_descriptions, strict=True))

