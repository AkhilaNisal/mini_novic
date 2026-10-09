import torch
import torch.nn.functional as F


# ============================================================
# Token decoding
# ============================================================

def decode_tokens(
    token_ids,
    id_to_token,
    eos_id,
    pad_id
):
    """
    Convert generated token IDs into a noun string.
    """

    words = []

    for token_id in token_ids:

        token_id = int(token_id)

        if token_id == eos_id:
            break

        if token_id == pad_id:
            continue

        words.append(
            id_to_token[token_id]
        )

    return " ".join(words)


# ============================================================
# Greedy generation
# ============================================================

@torch.no_grad()
def generate_greedy(
    model,
    embedding,
    eos_id,
    pad_id,
    max_length,
    device=None
):
    """
    Autoregressively generate a noun using greedy decoding.

    At each step, only the highest-probability next token
    is retained.
    """

    model.eval()

    if device is None:
        device = next(
            model.parameters()
        ).device

    if embedding.dim() == 1:

        embedding = embedding.unsqueeze(0)

    embedding = embedding.to(
        device
    )

    generated = []

    for _ in range(max_length):

        # ----------------------------------------------------
        # Empty token sequence for first prediction
        # ----------------------------------------------------

        if len(generated) == 0:

            token_inputs = torch.empty(
                (1, 0),
                dtype=torch.long,
                device=device
            )

        else:

            token_inputs = torch.tensor(
                [generated],
                dtype=torch.long,
                device=device
            )

        logits = model(
            embedding,
            token_inputs
        )

        next_token = (
            logits[
                0,
                -1,
                :
            ]
            .argmax()
            .item()
        )

        if next_token in (
            eos_id,
            pad_id
        ):
            break

        generated.append(
            next_token
        )

    return generated


# ============================================================
# Beam search
# ============================================================

@torch.no_grad()
def generate_beam_search(
    model,
    embedding,
    eos_id,
    pad_id,
    max_length,
    beam_width=5,
    return_top_k=None,
    device=None
):
    """
    Beam-search autoregressive generation.

    Each beam is represented as:

        (
            generated_token_ids,
            cumulative_log_probability,
            finished
        )
    """

    model.eval()

    if device is None:
        device = next(
            model.parameters()
        ).device

    if embedding.dim() == 1:

        embedding = embedding.unsqueeze(0)

    embedding = embedding.to(
        device
    )

    if return_top_k is None:

        return_top_k = beam_width

    beams = [
        (
            [],
            0.0,
            False
        )
    ]

    for _ in range(max_length):

        candidates = []

        for tokens, score, finished in beams:

            # ------------------------------------------------
            # Preserve already finished beams
            # ------------------------------------------------

            if finished:

                candidates.append(
                    (
                        tokens,
                        score,
                        True
                    )
                )

                continue

            # ------------------------------------------------
            # Current generated sequence
            # ------------------------------------------------

            if len(tokens) == 0:

                token_inputs = torch.empty(
                    (1, 0),
                    dtype=torch.long,
                    device=device
                )

            else:

                token_inputs = torch.tensor(
                    [tokens],
                    dtype=torch.long,
                    device=device
                )

            logits = model(
                embedding,
                token_inputs
            )

            next_logits = logits[
                0,
                -1,
                :
            ]

            log_probs = F.log_softmax(
                next_logits,
                dim=-1
            )

            # PAD should not be generated as a noun token.
            log_probs[
                pad_id
            ] = float("-inf")

            top_log_probs, top_ids = torch.topk(
                log_probs,
                k=min(
                    beam_width,
                    log_probs.size(0)
                )
            )

            # ------------------------------------------------
            # Expand current beam
            # ------------------------------------------------

            for token_log_prob, token_id in zip(
                top_log_probs.tolist(),
                top_ids.tolist()
            ):

                new_score = (
                    score
                    + token_log_prob
                )

                if token_id == eos_id:

                    candidates.append(
                        (
                            tokens.copy(),
                            new_score,
                            True
                        )
                    )

                else:

                    candidates.append(
                        (
                            tokens + [token_id],
                            new_score,
                            False
                        )
                    )

        # ----------------------------------------------------
        # Retain best sequences
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = candidates[
            :beam_width
        ]

        # Stop once all surviving beams reached EOS.
        if all(
            finished
            for _, _, finished in beams
        ):
            break

    beams.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return beams[
        :return_top_k
    ]