import torch
import torch.nn as nn


# ============================================================
# Attention mask
# ============================================================

def build_prefix_causal_mask(
    prefix_len: int,
    token_len: int,
    device
):
    """
    Construct the NOVIC-style self-attention mask.

    Prefix tokens:
        May freely attend to all other prefix tokens.

    Generated noun tokens:
        May attend to:
        - all prefix tokens
        - themselves
        - all previous noun tokens

    Parameters
    ----------
    prefix_len : int
        Number of prefix vectors.

    token_len : int
        Number of currently supplied noun tokens.

    device :
        Torch device.

    Returns
    -------
    torch.Tensor
        Attention mask of shape
        [prefix_len + token_len,
         prefix_len + token_len].
    """

    total_len = prefix_len + token_len

    mask = torch.full(
        (total_len, total_len),
        float("-inf"),
        device=device
    )

    # --------------------------------------------------------
    # Prefix tokens can freely communicate with each other
    # --------------------------------------------------------

    mask[
        :prefix_len,
        :prefix_len
    ] = 0.0

    # --------------------------------------------------------
    # Generated tokens use causal attention
    # --------------------------------------------------------

    for i in range(token_len):

        row = prefix_len + i

        mask[
            row,
            :prefix_len + i + 1
        ] = 0.0

    return mask


# ============================================================
# Original Mini-NOVIC baseline
# ============================================================

class MiniNOVICDecoder(nn.Module):
    """
    Original Mini-NOVIC decoder used in Notebooks 03-07.

    This class is preserved so that future experiments can
    compare the original baseline against the closer
    paper-style implementation.
    """

    def __init__(
        self,
        clip_dim: int,
        vocab_size: int,
        hidden_dim: int = 256,
        prefix_len: int = 4,
        max_seq_len: int = 4,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):

        super().__init__()

        self.clip_dim = clip_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_seq_len = max_seq_len

        # ----------------------------------------------------
        # CLIP embedding -> P prefix vectors
        # ----------------------------------------------------

        self.prefix_mapper = nn.Linear(
            clip_dim,
            prefix_len * hidden_dim
        )

        # ----------------------------------------------------
        # Target token embeddings
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

        # ----------------------------------------------------
        # Learned positional embeddings
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            prefix_len + max_seq_len,
            hidden_dim
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Hidden state -> vocabulary logits
        # ----------------------------------------------------

        self.output_projection = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(
        self,
        clip_embeddings,
        token_ids
    ):
        """
        Parameters
        ----------
        clip_embeddings:
            [B, clip_dim]

        token_ids:
            [B, T]

        Returns
        -------
        logits:
            [B, T + 1, vocab_size]

        The extra prediction comes from the final prefix token.

        Example:

            P1 P2 P3 P4 T1 T2 T3
                      ↓
                     P4 T1 T2 T3

        predicts:

                     T1 T2 T3 T4
        """

        batch_size = clip_embeddings.size(0)

        # ----------------------------------------------------
        # Create prefix vectors
        # ----------------------------------------------------

        prefix = self.prefix_mapper(
            clip_embeddings
        )

        prefix = prefix.view(
            batch_size,
            self.prefix_len,
            self.hidden_dim
        )

        # ----------------------------------------------------
        # Embed supplied noun tokens
        # ----------------------------------------------------

        token_vectors = self.token_embedding(
            token_ids
        )

        # ----------------------------------------------------
        # Prefix + noun tokens
        # ----------------------------------------------------

        x = torch.cat(
            [
                prefix,
                token_vectors
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Learned positions
        # ----------------------------------------------------

        positions = torch.arange(
            x.size(1),
            device=x.device
        )

        x = (
            x
            + self.position_embedding(
                positions
            ).unsqueeze(0)
        )

        # ----------------------------------------------------
        # NOVIC-style attention mask
        # ----------------------------------------------------

        attention_mask = build_prefix_causal_mask(
            self.prefix_len,
            token_ids.size(1),
            x.device
        )

        x = self.transformer(
            x,
            mask=attention_mask
        )

        # ----------------------------------------------------
        # Last prefix predicts first noun token
        # ----------------------------------------------------

        prediction_states = x[
            :,
            self.prefix_len - 1:,
            :
        ]

        logits = self.output_projection(
            prediction_states
        )

        return logits


# ============================================================
# Closer paper-style NOVIC decoder
# ============================================================

class PaperStyleNOVICDecoder(nn.Module):
    """
    Mini-scale decoder incorporating more architectural
    choices described in the NOVIC paper.

    Main differences from MiniNOVICDecoder:

    - bias-free prefix projection
    - bias-free Transformer linear/LayerNorm layers
    - contractive feed-forward dimension
    - positional dropout
    - tied input/output token weights
    - bias-free output projection
    """

    def __init__(
        self,
        clip_dim: int,
        vocab_size: int,
        hidden_dim: int = 256,
        prefix_len: int = 4,
        max_seq_len: int = 4,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1
    ):

        super().__init__()

        self.clip_dim = clip_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_seq_len = max_seq_len
        self.ff_dim = ff_dim

        # ----------------------------------------------------
        # Bias-free embedding -> prefix projection
        # ----------------------------------------------------

        self.prefix_mapper = nn.Linear(
            clip_dim,
            prefix_len * hidden_dim,
            bias=False
        )

        # ----------------------------------------------------
        # Learned vocabulary embeddings
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

        # ----------------------------------------------------
        # Learned positional embeddings
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            prefix_len + max_seq_len,
            hidden_dim
        )

        self.position_dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Decoder-only Transformer implemented with
        # self-attention encoder blocks + custom mask
        # ----------------------------------------------------

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,

            # Contractive FFN
            dim_feedforward=ff_dim,

            dropout=dropout,
            activation="gelu",

            batch_first=True,

            # Pre-LayerNorm
            norm_first=True,

            # Removes biases from linear + LayerNorm modules
            bias=False
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Bias-free vocabulary projection
        # ----------------------------------------------------

        self.output_projection = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # Weight tying
        #
        # Input token embedding and output vocabulary
        # projection use the same weight matrix.
        # ----------------------------------------------------

        self.output_projection.weight = (
            self.token_embedding.weight
        )

    def forward(
        self,
        clip_embeddings,
        token_ids
    ):

        batch_size = clip_embeddings.size(0)

        # ----------------------------------------------------
        # CLIP embedding -> prefix vectors
        # ----------------------------------------------------

        prefix = self.prefix_mapper(
            clip_embeddings
        )

        prefix = prefix.view(
            batch_size,
            self.prefix_len,
            self.hidden_dim
        )

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_vectors = self.token_embedding(
            token_ids
        )

        # ----------------------------------------------------
        # Build transformer sequence
        # ----------------------------------------------------

        x = torch.cat(
            [
                prefix,
                token_vectors
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Add learned positional embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            x.size(1),
            device=x.device
        )

        position_vectors = self.position_embedding(
            positions
        ).unsqueeze(0)

        x = x + position_vectors

        # Paper-style positional dropout
        x = self.position_dropout(
            x
        )

        # ----------------------------------------------------
        # Prefix-full + noun-causal attention
        # ----------------------------------------------------

        attention_mask = build_prefix_causal_mask(
            self.prefix_len,
            token_ids.size(1),
            x.device
        )

        x = self.transformer(
            x,
            mask=attention_mask
        )

        # ----------------------------------------------------
        # No BOS token:
        #
        # P_last -> target token 1
        # T1     -> target token 2
        # T2     -> target token 3
        # ...
        # ----------------------------------------------------

        prediction_states = x[
            :,
            self.prefix_len - 1:,
            :
        ]

        logits = self.output_projection(
            prediction_states
        )

        return logits