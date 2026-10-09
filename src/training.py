import torch
import torch.nn.functional as F


# ============================================================
# Decoder loss
# ============================================================

def compute_decoder_loss(
    model,
    clip_embeddings,
    target_tokens,
    pad_id
):
    """
    Compute shifted autoregressive cross-entropy loss.

    target_tokens:
        [B, L]

    teacher input:
        target_tokens[:, :-1]

    Model internally returns:
        final-prefix state + token states

    Therefore logits already align with the entire target
    sequence and must NOT be sliced again.
    """

    token_inputs = (
        target_tokens[
            :,
            :-1
        ]
    )

    logits = model(
        clip_embeddings,
        token_inputs
    )

    if logits.shape[:2] != target_tokens.shape:

        raise RuntimeError(
            "Prediction/target sequence mismatch: "
            f"logits={logits.shape}, "
            f"targets={target_tokens.shape}"
        )

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1)
        ),
        target_tokens.reshape(
            -1
        ),
        ignore_index=pad_id
    )

    return loss, logits


# ============================================================
# Token accuracy
# ============================================================

@torch.no_grad()
def token_accuracy(
    logits,
    targets,
    pad_id
):
    """
    Calculate token accuracy ignoring PAD positions.
    """

    predictions = logits.argmax(
        dim=-1
    )

    valid_mask = (
        targets != pad_id
    )

    correct = (
        predictions == targets
    ) & valid_mask

    denominator = valid_mask.sum()

    if denominator.item() == 0:
        return 0.0

    return (
        correct.sum().float()
        / denominator.float()
    ).item()


# ============================================================
# Training epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    pad_id,
    noise_function=None,
    grad_clip=1.0
):
    """
    Train for one epoch.

    Expected dataloader batch:

        embeddings, target_tokens
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0

    for embeddings, targets in dataloader:

        embeddings = embeddings.to(
            device
        )

        targets = targets.to(
            device
        )

        # ----------------------------------------------------
        # Optional embedding noise
        # ----------------------------------------------------

        if noise_function is not None:

            embeddings = noise_function(
                embeddings
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, logits = compute_decoder_loss(
            model,
            embeddings,
            targets,
            pad_id
        )

        loss.backward()

        if grad_clip is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

        optimizer.step()

        batch_size = embeddings.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=-1
        )

        valid_mask = (
            targets != pad_id
        )

        total_correct += (
            (
                predictions == targets
            )
            & valid_mask
        ).sum().item()

        total_valid_tokens += (
            valid_mask.sum().item()
        )

    dataset_size = len(
        dataloader.dataset
    )

    average_loss = (
        total_loss
        / dataset_size
    )

    accuracy = (
        total_correct
        / total_valid_tokens
        if total_valid_tokens > 0
        else 0.0
    )

    return {
        "loss": average_loss,
        "token_accuracy": accuracy
    }


# ============================================================
# Validation epoch
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    device,
    pad_id
):
    """
    Validate using clean embeddings.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_valid_tokens = 0

    for embeddings, targets in dataloader:

        embeddings = embeddings.to(
            device
        )

        targets = targets.to(
            device
        )

        loss, logits = compute_decoder_loss(
            model,
            embeddings,
            targets,
            pad_id
        )

        batch_size = embeddings.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = logits.argmax(
            dim=-1
        )

        valid_mask = (
            targets != pad_id
        )

        total_correct += (
            (
                predictions == targets
            )
            & valid_mask
        ).sum().item()

        total_valid_tokens += (
            valid_mask.sum().item()
        )

    dataset_size = len(
        dataloader.dataset
    )

    average_loss = (
        total_loss
        / dataset_size
    )

    accuracy = (
        total_correct
        / total_valid_tokens
        if total_valid_tokens > 0
        else 0.0
    )

    return {
        "loss": average_loss,
        "token_accuracy": accuracy
    }