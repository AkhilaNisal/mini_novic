import numpy as np
import pandas as pd
import torch

from .decoding import (
    generate_greedy,
    generate_beam_search,
    decode_tokens
)


# ============================================================
# Text generation evaluation
# ============================================================

@torch.no_grad()
def evaluate_text_generation(
    model,
    embeddings,
    expected_nouns,
    id_to_token,
    eos_id,
    pad_id,
    max_length,
    device=None
):
    """
    Evaluate exact greedy noun generation from embeddings.
    """

    if device is None:
        device = next(
            model.parameters()
        ).device

    records = []

    for index in range(
        len(embeddings)
    ):

        embedding = embeddings[
            index
        ]

        token_ids = generate_greedy(
            model=model,
            embedding=embedding,
            eos_id=eos_id,
            pad_id=pad_id,
            max_length=max_length,
            device=device
        )

        generated = decode_tokens(
            token_ids,
            id_to_token=id_to_token,
            eos_id=eos_id,
            pad_id=pad_id
        )

        expected = str(
            expected_nouns[index]
        )

        records.append({
            "index": index,
            "expected": expected,
            "generated": generated,
            "correct": (
                generated == expected
            )
        })

    results = pd.DataFrame(
        records
    )

    accuracy = (
        results[
            "correct"
        ].mean()
    )

    return accuracy, results


# ============================================================
# Image generation evaluation
# ============================================================

@torch.no_grad()
def evaluate_images(
    model,
    image_df,
    encode_image_fn,
    id_to_token,
    eos_id,
    pad_id,
    max_length,
    image_path_col="image_path",
    expected_col="expected",
    device=None
):
    """
    Evaluate greedy noun generation on an image dataframe.

    `encode_image_fn(path)` must return a normalized embedding.
    """

    if device is None:
        device = next(
            model.parameters()
        ).device

    records = []

    for _, row in image_df.iterrows():

        embedding = encode_image_fn(
            row[
                image_path_col
            ]
        )

        token_ids = generate_greedy(
            model=model,
            embedding=embedding,
            eos_id=eos_id,
            pad_id=pad_id,
            max_length=max_length,
            device=device
        )

        generated = decode_tokens(
            token_ids,
            id_to_token=id_to_token,
            eos_id=eos_id,
            pad_id=pad_id
        )

        expected = str(
            row[
                expected_col
            ]
        )

        record = row.to_dict()

        record[
            "generated"
        ] = generated

        record[
            "correct"
        ] = (
            generated == expected
        )

        records.append(
            record
        )

    results = pd.DataFrame(
        records
    )

    accuracy = results[
        "correct"
    ].mean()

    return accuracy, results


# ============================================================
# Beam-search image evaluation
# ============================================================

@torch.no_grad()
def evaluate_beam_images(
    model,
    image_df,
    encode_image_fn,
    id_to_token,
    eos_id,
    pad_id,
    max_length,
    beam_width=5,
    image_path_col="image_path",
    expected_col="expected",
    device=None
):
    """
    Evaluate Top-1 beam-search output.
    """

    if device is None:
        device = next(
            model.parameters()
        ).device

    records = []

    for _, row in image_df.iterrows():

        embedding = encode_image_fn(
            row[
                image_path_col
            ]
        )

        beams = generate_beam_search(
            model=model,
            embedding=embedding,
            eos_id=eos_id,
            pad_id=pad_id,
            max_length=max_length,
            beam_width=beam_width,
            return_top_k=1,
            device=device
        )

        tokens, score, finished = beams[
            0
        ]

        generated = decode_tokens(
            tokens,
            id_to_token=id_to_token,
            eos_id=eos_id,
            pad_id=pad_id
        )

        expected = str(
            row[
                expected_col
            ]
        )

        record = row.to_dict()

        record.update({
            "generated": generated,
            "log_probability": score,
            "finished": finished,
            "correct": (
                generated == expected
            )
        })

        records.append(
            record
        )

    results = pd.DataFrame(
        records
    )

    accuracy = results[
        "correct"
    ].mean()

    return accuracy, results


# ============================================================
# Top-k candidate generation
# ============================================================

@torch.no_grad()
def generate_topk_image_predictions(
    model,
    image_df,
    encode_image_fn,
    id_to_token,
    eos_id,
    pad_id,
    max_length,
    k=5,
    image_path_col="image_path",
    expected_col="expected",
    device=None
):
    """
    Generate the top-k beam candidates for every image.
    """

    if device is None:
        device = next(
            model.parameters()
        ).device

    records = []

    for _, row in image_df.iterrows():

        embedding = encode_image_fn(
            row[
                image_path_col
            ]
        )

        beams = generate_beam_search(
            model=model,
            embedding=embedding,
            eos_id=eos_id,
            pad_id=pad_id,
            max_length=max_length,
            beam_width=k,
            return_top_k=k,
            device=device
        )

        record = {
            "image_path":
                row[
                    image_path_col
                ],

            "expected":
                row[
                    expected_col
                ]
        }

        for rank, (
            tokens,
            score,
            finished
        ) in enumerate(
            beams,
            start=1
        ):

            record[
                f"rank_{rank}"
            ] = decode_tokens(
                tokens,
                id_to_token=id_to_token,
                eos_id=eos_id,
                pad_id=pad_id
            )

            record[
                f"score_{rank}"
            ] = score

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# Top-k accuracy
# ============================================================

def calculate_topk_accuracy(
    results,
    k
):
    """
    Compute whether expected noun occurs within first k ranks.
    """

    correct = []

    for _, row in results.iterrows():

        predictions = [
            row[
                f"rank_{rank}"
            ]
            for rank in range(
                1,
                k + 1
            )
        ]

        correct.append(
            row["expected"]
            in predictions
        )

    return float(
        np.mean(
            correct
        )
    )