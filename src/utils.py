import random
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed=42
):
    """
    Set Python, NumPy, and PyTorch random seeds.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Parameter counting
# ============================================================

def count_parameters(
    model
):
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable
    }


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    extra=None
):
    """
    Save model and optional optimizer/training metadata.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict()
    }

    if optimizer is not None:

        checkpoint[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

    if epoch is not None:

        checkpoint[
            "epoch"
        ] = epoch

    if extra is not None:

        checkpoint.update(
            extra
        )

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    path,
    model=None,
    optimizer=None,
    device="cpu"
):
    """
    Load checkpoint and optionally restore model/optimizer.
    """

    checkpoint = torch.load(
        path,
        map_location=device
    )

    if model is not None:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    if (
        optimizer is not None
        and
        "optimizer_state_dict"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint