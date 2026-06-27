from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lunar_icenav.features.texture import robust_normalize


CHANNEL_NAMES = [
    "sar_intensity_log",
    "lh_lv_ratio_proxy",
    "texture_roughness_proxy",
    "polarization_imbalance_proxy",
    "rule_candidate_score",
]


def run_unet_prototype(features: dict[str, np.ndarray], pseudo_label: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        return fallback_prediction(features, pseudo_label, f"torch unavailable: {exc}")

    unet_cfg = config.get("unet", {})
    if not unet_cfg.get("enabled", True):
        return fallback_prediction(features, pseudo_label, "disabled in config")

    valid = features["valid"].astype(bool)
    x = build_input_stack(features, valid)
    y = pseudo_label.astype("float32")[None, :, :]
    tile_size = int(unet_cfg.get("tile_size", 128))
    max_tiles = int(unet_cfg.get("max_tiles", 64))
    tiles = make_spatial_tiles(x, y, tile_size, max_tiles)
    if len(tiles) < 2 or y.sum() < 5:
        return fallback_prediction(features, pseudo_label, "not enough pseudo-label signal for training")

    train_tiles, val_tiles = spatial_train_val_split(tiles, float(unet_cfg.get("validation_fraction", 0.25)))
    train_x, train_y = augment_training_tiles(train_tiles)
    val_x, val_y = stack_tiles(val_tiles)
    if train_x.size == 0 or val_x.size == 0:
        return fallback_prediction(features, pseudo_label, "spatial split did not produce train/validation tiles")

    torch.manual_seed(7)
    device = torch.device("cpu")
    model = TinyUNet(in_channels=train_x.shape[1]).to(device)
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_dataset, batch_size=int(unet_cfg.get("batch_size", 4)), shuffle=True)
    val_tensor_x = torch.from_numpy(val_x).to(device)
    val_tensor_y = torch.from_numpy(val_y).to(device)

    pos = float(train_y.sum())
    neg = float(train_y.size - pos)
    pos_weight = torch.tensor([min(max(neg / max(pos, 1.0), 1.0), 500.0)], dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(unet_cfg.get("learning_rate", 0.001)))

    history: list[dict[str, float]] = []
    for epoch in range(1, int(unet_cfg.get("epochs", 12)) + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            target = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = segmentation_loss(logits, target, pos_weight, F)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor_x)
            val_loss = float(segmentation_loss(val_logits, val_tensor_y, pos_weight, F).item())
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_pred = threshold_like_labels(val_prob[:, 0], val_y[:, 0])
        val_metrics = agreement_metrics(val_pred, val_y[:, 0].astype(bool))
        history.append({
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
            "val_loss": val_loss,
            "val_pseudo_iou": float(val_metrics["pseudo_iou"]),
            "val_pseudo_dice": float(val_metrics["pseudo_dice"]),
        })

    model.eval()
    with torch.no_grad():
        full = torch.from_numpy(x[None, ...]).to(device)
        logits = model(full).cpu().numpy()[0, 0]
    prob = np.nan_to_num(1.0 / (1.0 + np.exp(-logits)), nan=0.0, posinf=1.0, neginf=0.0)
    label_fraction = float(np.clip(pseudo_label.mean(), 0.001, 0.20))
    threshold = float(np.nanquantile(prob, 1.0 - label_fraction))
    prediction = prob >= threshold
    metrics = agreement_metrics(prediction, pseudo_label)
    metrics.update({
        "training_loss_last": history[-1]["train_loss"] if history else np.nan,
        "validation_loss_last": history[-1]["val_loss"] if history else np.nan,
        "validation_pseudo_iou_last": history[-1]["val_pseudo_iou"] if history else np.nan,
        "validation_pseudo_dice_last": history[-1]["val_pseudo_dice"] if history else np.nan,
        "training_tiles": int(len(train_tiles)),
        "validation_tiles": int(len(val_tiles)),
        "augmented_training_tiles": int(train_x.shape[0]),
        "prediction_threshold": threshold,
    })

    checkpoint_path = Path(unet_cfg.get("checkpoint_path", "models/unet_pseudolabel.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "channel_names": CHANNEL_NAMES,
        "config": unet_cfg,
        "metrics": metrics,
        "note": "Pseudo-label checkpoint for prototype agreement experiments only.",
    }, checkpoint_path)

    return {
        "prediction_probability": prob.astype("float32"),
        "prediction_mask": prediction.astype(bool),
        "metrics": metrics,
        "history": pd.DataFrame(history),
        "tile_inventory": pd.DataFrame(tile_records(train_tiles, "train") + tile_records(val_tiles, "validation")),
        "channel_names": CHANNEL_NAMES,
        "checkpoint_path": str(checkpoint_path),
        "model_summary": "TinyUNet: 5-channel input -> 12/24 feature encoder -> skip-connected decoder -> binary pseudo-label mask.",
        "augmentation_note": "Training tiles include original, horizontal flip, vertical flip, and 90-degree rotation augmentations. Spatial coordinates are not preserved for augmented tiles.",
        "note": "weakly supervised U-Net trained against rule-based pseudo-labels and SAR screening features; metrics measure pseudo-label agreement, not ground-truth ice accuracy",
    }


def build_input_stack(features: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    channels = [
        robust_normalize(features["intensity"], valid),
        robust_normalize(features["cpr_style_ratio_proxy"], valid),
        robust_normalize(features["texture"], valid),
        robust_normalize(features["polarization_imbalance_proxy"], valid),
        robust_normalize(features["candidate_score"], valid),
    ]
    return np.nan_to_num(np.stack(channels, axis=0).astype("float32"), nan=0.0, posinf=1.0, neginf=0.0)


def segmentation_loss(logits, target, pos_weight, F) -> Any:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob_batch = F.sigmoid(logits)
    inter = (prob_batch * target).sum(dim=(1, 2, 3))
    denom = prob_batch.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice_loss = 1.0 - ((2.0 * inter + 1e-6) / denom).mean()
    return bce + dice_loss


class TinyUNet:  # replaced with nn.Module subclass at import time
    pass


def _build_tiny_unet_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _TinyUNet(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 12, 3, padding=1), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
            self.dec1 = nn.Sequential(nn.Conv2d(36, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, 2))
            up = F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(torch.cat([up, e1], dim=1))
            return self.out(d1)

    return _TinyUNet


TinyUNet = _build_tiny_unet_class()


def make_spatial_tiles(x: np.ndarray, y: np.ndarray, tile: int, max_tiles: int) -> list[dict[str, Any]]:
    _, h, w = x.shape
    stride = max(tile // 2, 1)
    tiles: list[dict[str, Any]] = []
    for r in range(0, max(1, h - tile + 1), stride):
        for c in range(0, max(1, w - tile + 1), stride):
            xx = x[:, r:r + tile, c:c + tile]
            yy = y[:, r:r + tile, c:c + tile]
            if xx.shape[-2:] != (tile, tile):
                continue
            tiles.append({
                "x": xx,
                "y": yy,
                "row": r,
                "col": c,
                "positive_pixels": int(yy.sum()),
                "positive_fraction": float(yy.mean()),
            })
    positives = [t for t in tiles if t["positive_pixels"] > 0]
    negatives = [t for t in tiles if t["positive_pixels"] == 0]
    positives.sort(key=lambda t: t["positive_pixels"], reverse=True)
    # Keep enough background tiles to avoid a model that predicts everything positive.
    selected = positives[: max_tiles // 2] + negatives[: max(0, max_tiles - min(len(positives), max_tiles // 2))]
    if len(selected) < min(max_tiles, len(tiles)):
        selected_ids = {id(t) for t in selected}
        selected += [t for t in tiles if id(t) not in selected_ids][: max_tiles - len(selected)]
    return selected[:max_tiles]


def spatial_train_val_split(tiles: list[dict[str, Any]], validation_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(tiles) < 3:
        return tiles[:1], tiles[1:]
    cols = np.array([t["col"] for t in tiles])
    split_col = float(np.quantile(cols, 1.0 - validation_fraction))
    train = [t for t in tiles if t["col"] < split_col]
    val = [t for t in tiles if t["col"] >= split_col]
    if not val or sum(t["positive_pixels"] for t in val) == 0:
        positives = [t for t in tiles if t["positive_pixels"] > 0]
        val = positives[-max(1, len(positives) // 4):] if positives else tiles[-max(1, len(tiles) // 4):]
        val_ids = {id(t) for t in val}
        train = [t for t in tiles if id(t) not in val_ids]
    if not train:
        train, val = tiles[:-1], tiles[-1:]
    return train, val


def augment_training_tiles(tiles: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for tile in tiles:
        x = tile["x"]
        y = tile["y"]
        variants = [
            (x, y),
            (np.flip(x, axis=2).copy(), np.flip(y, axis=2).copy()),
            (np.flip(x, axis=1).copy(), np.flip(y, axis=1).copy()),
            (np.rot90(x, k=1, axes=(1, 2)).copy(), np.rot90(y, k=1, axes=(1, 2)).copy()),
        ]
        for xx, yy in variants:
            xs.append(xx)
            ys.append(yy)
    if not xs:
        return np.empty((0, 5, 1, 1), dtype="float32"), np.empty((0, 1, 1, 1), dtype="float32")
    return np.stack(xs).astype("float32"), np.stack(ys).astype("float32")


def stack_tiles(tiles: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not tiles:
        return np.empty((0, 5, 1, 1), dtype="float32"), np.empty((0, 1, 1, 1), dtype="float32")
    return np.stack([t["x"] for t in tiles]).astype("float32"), np.stack([t["y"] for t in tiles]).astype("float32")


def threshold_like_labels(prob_tiles: np.ndarray, label_tiles: np.ndarray) -> np.ndarray:
    label_fraction = float(np.clip(label_tiles.mean(), 0.001, 0.20))
    threshold = float(np.nanquantile(prob_tiles, 1.0 - label_fraction))
    return prob_tiles >= threshold


def tile_records(tiles: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [{
        "split": split,
        "row": int(tile["row"]),
        "col": int(tile["col"]),
        "positive_pixels": int(tile["positive_pixels"]),
        "positive_fraction": float(tile["positive_fraction"]),
    } for tile in tiles]


def agreement_metrics(pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    label = label.astype(bool)
    inter = float(np.logical_and(pred, label).sum())
    union = float(np.logical_or(pred, label).sum())
    denom = float(pred.sum() + label.sum())
    return {
        "pseudo_iou": inter / union if union else 0.0,
        "pseudo_dice": 2 * inter / denom if denom else 0.0,
        "prediction_fraction": float(pred.mean()),
        "pseudo_label_fraction": float(label.mean()),
    }


def fallback_prediction(features: dict[str, np.ndarray], pseudo_label: np.ndarray, reason: str) -> dict[str, Any]:
    prob = robust_normalize(features["candidate_score"], features["valid"].astype(bool))
    pred = prob >= np.nanquantile(prob[np.isfinite(prob)], 0.88)
    metrics = agreement_metrics(pred, pseudo_label)
    return {
        "prediction_probability": prob.astype("float32"),
        "prediction_mask": pred.astype(bool),
        "metrics": metrics,
        "history": pd.DataFrame(),
        "tile_inventory": pd.DataFrame(),
        "channel_names": CHANNEL_NAMES,
        "checkpoint_path": "",
        "model_summary": "Fallback candidate-score threshold; U-Net did not train.",
        "augmentation_note": "No augmentation used in fallback.",
        "note": f"U-Net proof-of-concept fallback used: {reason}. This is pseudo-label agreement, not real accuracy.",
    }
