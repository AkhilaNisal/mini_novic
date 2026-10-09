import torch
import torch.nn.functional as F


# ============================================================
# Gaussian embedding noise
# ============================================================

def add_gaussian_noise(
    embeddings,
    sigma=0.05
):
    """
    Add elementwise Gaussian noise and renormalize
    embeddings to unit length.
    """

    if sigma == 0:

        return embeddings

    noise = (
        sigma
        * torch.randn_like(
            embeddings
        )
    )

    noisy_embeddings = (
        embeddings
        + noise
    )

    return F.normalize(
        noisy_embeddings,
        dim=-1
    )


# ============================================================
# Controlled angular noise
# ============================================================

def add_angular_noise(
    embeddings,
    min_angle_deg=45.0,
    max_angle_deg=75.0
):
    """
    Rotate each unit embedding in a random orthogonal
    direction by an angle sampled uniformly from the
    requested angular range.
    """

    embeddings = F.normalize(
        embeddings,
        dim=-1
    )

    random_vectors = torch.randn_like(
        embeddings
    )

    # Remove component parallel to original embedding.
    projections = (
        random_vectors
        * embeddings
    ).sum(
        dim=-1,
        keepdim=True
    )

    orthogonal = (
        random_vectors
        - projections * embeddings
    )

    orthogonal = F.normalize(
        orthogonal,
        dim=-1
    )

    angles_deg = torch.empty(
        (
            embeddings.size(0),
            1
        ),
        device=embeddings.device,
        dtype=embeddings.dtype
    ).uniform_(
        min_angle_deg,
        max_angle_deg
    )

    angles_rad = torch.deg2rad(
        angles_deg
    )

    noisy_embeddings = (
        torch.cos(
            angles_rad
        )
        * embeddings
        +
        torch.sin(
            angles_rad
        )
        * orthogonal
    )

    return F.normalize(
        noisy_embeddings,
        dim=-1
    )


# ============================================================
# Mixed NOVIC-style noise
# ============================================================

def add_novic_style_noise(
    embeddings,
    gaussian_sigma=0.05,
    gaussian_probability=0.85,
    min_angle_deg=45.0,
    max_angle_deg=75.0
):
    """
    Apply Gaussian noise to a fraction of embeddings and
    angular noise to the remaining embeddings.

    Default:
        85% Gaussian
        15% angular
    """

    batch_size = embeddings.size(
        0
    )

    gaussian_mask = (
        torch.rand(
            batch_size,
            device=embeddings.device
        )
        <
        gaussian_probability
    )

    output = embeddings.clone()

    # --------------------------------------------------------
    # Gaussian subset
    # --------------------------------------------------------

    if gaussian_mask.any():

        output[
            gaussian_mask
        ] = add_gaussian_noise(
            embeddings[
                gaussian_mask
            ],
            sigma=gaussian_sigma
        )

    # --------------------------------------------------------
    # Angular subset
    # --------------------------------------------------------

    angular_mask = (
        ~gaussian_mask
    )

    if angular_mask.any():

        output[
            angular_mask
        ] = add_angular_noise(
            embeddings[
                angular_mask
            ],
            min_angle_deg=min_angle_deg,
            max_angle_deg=max_angle_deg
        )

    return F.normalize(
        output,
        dim=-1
    )


# ============================================================
# Angular distance
# ============================================================

def angular_distance_degrees(
    a,
    b
):
    """
    Compute angular distance between embedding vectors.
    """

    a = F.normalize(
        a,
        dim=-1
    )

    b = F.normalize(
        b,
        dim=-1
    )

    cosine_similarity = (
        a * b
    ).sum(
        dim=-1
    )

    cosine_similarity = torch.clamp(
        cosine_similarity,
        -1.0,
        1.0
    )

    return torch.rad2deg(
        torch.acos(
            cosine_similarity
        )
    )