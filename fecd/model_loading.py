"""Safe BoxPromptIML construction with an explicit local TinyViT checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_boxpromptiml_with_local_tinyvit(
    checkpoint_path: str | Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the unchanged student while preventing TinyViT URL downloads.

    ``boxpromptiml.BoxPromptIML`` constructs its backbone with
    ``pretrained=True``. The upstream TinyViT helper maps that flag to a GitHub
    URL. This adapter temporarily replaces only the factory symbol imported by
    ``boxpromptiml`` with a factory that creates ``pretrained=False`` and loads
    the supplied local state dict strictly. No project source is edited.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    from TinyViT.models.tiny_vit import tiny_vit_5m_224
    import boxpromptiml

    load_info: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "factory_pretrained_requested_by_model": True,
        "network_download_used": False,
    }

    def local_factory(pretrained: bool = False, **kwargs: Any) -> torch.nn.Module:
        if not pretrained:
            raise RuntimeError("BoxPromptIML unexpectedly requested an unpretrained TinyViT")
        model = tiny_vit_5m_224(pretrained=False, **kwargs)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError("TinyViT checkpoint must contain a top-level 'model' state dict")
        state = payload["model"]
        if not isinstance(state, dict):
            raise RuntimeError("TinyViT checkpoint 'model' entry is not a state dict")
        state = {key: value for key, value in state.items() if not key.endswith("attention_bias_idxs")}
        incompatible = model.load_state_dict(state, strict=True)
        load_info["loaded_tensor_count"] = len(state)
        load_info["missing_keys"] = list(incompatible.missing_keys)
        load_info["unexpected_keys"] = list(incompatible.unexpected_keys)
        load_info["strict_load"] = True
        return model

    original_factory = boxpromptiml.tiny_vit_5m_224
    boxpromptiml.tiny_vit_5m_224 = local_factory
    try:
        model = boxpromptiml.BoxPromptIML()
    finally:
        boxpromptiml.tiny_vit_5m_224 = original_factory

    return model, load_info
