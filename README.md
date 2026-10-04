# Mini-NOVIC

A staged reproduction and implementation of **Unconstrained Open Vocabulary Image
Classification: Zero-Shot Transfer from Text to Image via CLIP Inversion**, by
**Philipp Allgeuer, Kyra Ahrens, and Stefan Wermter** (WACV 2025).

Official source code: [pallgeuer/novic](https://github.com/pallgeuer/novic).

## 1. Project Overview

NOVIC is an unconstrained open vocabulary image classifier that generates object
names from image embeddings. Mini-NOVIC is a university computer vision paper
reproduction project that develops this idea through small, inspectable experiments.

Standard CLIP classification normally requires a candidate label list:

```text
image → CLIP image encoder → image embedding
      → compare with candidate text embeddings → choose one candidate
```

It can only select from supplied candidates. NOVIC instead learns an object decoder:

```text
image → frozen CLIP image encoder → image embedding
      → embedding-to-prefix projection → autoregressive Transformer decoder
      → generated object noun
```

The key idea is zero-shot transfer from text to image:

```text
Training:
text → frozen CLIP text encoder → text embedding → decoder → object noun

Inference:
image → frozen CLIP image encoder → image embedding → same decoder → object noun
```

The decoder is trained from text-derived embeddings rather than image-label pairs.
CLIP's shared semantic space supports this transfer, although text embeddings and
image embeddings are not identical. Label generation removes the need for a
user-supplied candidate list; recognition still depends on learned representations
and training coverage.

## 2. Main Objectives

**Goal A — Build an understandable Mini-NOVIC implementation.** Develop:

- CLIP image/text embeddings, an object noun dataset, and caption generation.
- Frozen text embeddings, embedding caching, and embedding noise.
- Embedding-to-prefix projection and an autoregressive Transformer.
- Multi-token noun generation, EOS handling, greedy decoding, and beam search.
- Zero-shot image transfer, evaluation, and ablation experiments.

**Goal B — Reproduce part of the paper results using official NOVIC.** Later, use
its official implementation and released checkpoints to reproduce at least part
of the published results and compare them with reference values.

These are separate tracks: understanding/implementation and direct paper
reproduction. Decoder training and official benchmark reproduction remain pending.

## 3. Why a Staged Implementation?

The original work operates at much larger scale: tens of thousands of object nouns,
millions of synthetic text samples, large embedding caches, template/multiset/LLM
text generation, larger CLIP/SigLIP models, substantial decoder training, beam
search, and benchmark evaluation.

Mini-NOVIC uses a smaller controlled setup before scaling toward the full paper
configuration. The first scientific goal is to establish the transfer mechanism,
rather than maximize accuracy:

```text
decoder trained on text embeddings
  → accepts image embeddings at inference
  → generates meaningful object nouns

small working system → verify components → add complexity → scale gradually
```

## 4. Planned Architecture

Training pipeline:

```text
caption
  → frozen CLIP text encoder
  → L2-normalized text embedding
  → optional embedding noise
  → prefix projection
  → P prefix vectors
  → decoder-only Transformer
  → noun tokens
  → EOS
```

Inference pipeline:

```text
image
  → frozen CLIP image encoder
  → L2-normalized image embedding
  → same prefix projection
  → same Transformer decoder
  → generated noun
```

The image path is introduced only at inference in the core zero-shot transfer
experiment. CLIP stays frozen; the prefix projection and decoder are trainable.

## 5. Development Workflow

- **Local:** VS Code for source code, notebooks, configs, README, and small selected
  results; Git and GitHub for version control.
- **Cloud:** Google Colab for GPU execution, PyTorch, OpenCLIP, training, model
  downloads, and embedding generation.
- **Google Drive:** persistent large datasets, embedding caches, checkpoints,
  large result files, and heavy outputs, rather than GitHub.

```text
VS Code → GitHub → Google Colab → Google Drive for large persistent artifacts
```

GitHub is the source-code/version-control layer. Drive retains heavy artifacts
across Colab sessions. Small selected datasets and result CSVs can remain in GitHub.

`configs/clip_exploration.yaml` records seed `42`, OpenCLIP model `ViT-B-32`,
pretrained identifier `laion2b_s34b_b79k`, and a CUDA runtime. Its Drive paths are:

```text
/content/drive/MyDrive/mini_novic_storage/
├── datasets/
├── embeddings/
├── checkpoints/
└── results/
```

To run the existing experiments, open the notebooks in Colab, select a GPU runtime,
run their setup cells, mount Drive, and populate the expected dataset paths.
`requirements-cloud.txt` lists cloud dependencies; verify PyTorch/CUDA availability
in the runtime. `requirements.txt` is currently an empty placeholder. A standalone
decoder training command is not yet available.

## 6. Repository Structure

```text
mini_novic/
├── notebooks/
│   ├── 01_clip_exploration_colab.ipynb
│   └── 02_text_embedding_dataset.ipynb
├── src/                              # Reusable implementation placeholder
├── configs/
│   └── clip_exploration.yaml
├── data/
│   ├── clip_test/                     # Selected CLIP test images
│   ├── test_images/                   # Further image-test placeholder
│   └── text_embedding_dataset/
│       └── mini_novic_captions.csv
├── results/
│   ├── clip_exploration/              # Selected similarity CSVs
│   └── text_embedding_dataset/        # Baseline history and predictions
├── embeddings/                       # Ignored local artifact directory
├── checkpoints/                      # Ignored local artifact directory
├── requirements.txt
├── requirements-cloud.txt
├── .gitignore
└── README.md
```

`embeddings/` and `checkpoints/` are excluded by `.gitignore` and are not intended
for GitHub. Cloud artifacts live in Drive. Notebook 03 is planned and does not yet
exist in the repository.

## 7. Completed Work

### Milestone 0 — Setup

**Status: Completed.**

- Local repository structure and Git initialization.
- GitHub connection and version-control workflow.
- Google Colab GPU workflow and Google Drive persistent storage.
- Source/config/result separation.

### Notebook 01 — CLIP Exploration

File: [notebooks/01_clip_exploration_colab.ipynb](notebooks/01_clip_exploration_colab.ipynb)

**Status: Completed.** This notebook explored:

- Image preprocessing and text tokenization.
- CLIP image/text encoders and embedding dimensions.
- L2 normalization, cosine similarity, and zero-shot classification.
- Candidate-label limitations and prompt sensitivity.
- Image-text similarity matrices and semantic relationships in text embedding space.
- PCA visualization and the image-text modality gap.

```text
image → CLIP image embedding
      → compare with candidate text embeddings → choose highest-similarity label
```

Changing the candidate list can change the result. Removing the correct label
prevents its selection, motivating NOVIC's generative decoder.

The modality-gap experiment shows that matching image and text embeddings are
semantically aligned but not identical. This matters when a text-trained decoder
receives image embeddings. Selected CSVs are in `results/clip_exploration/`.

### Notebook 02 — Text Embedding Dataset and Linear Baseline

File: [notebooks/02_text_embedding_dataset.ipynb](notebooks/02_text_embedding_dataset.ipynb)

**Status: Completed.** This notebook created:

- A controlled vocabulary of 50 visually concrete nouns, including multi-word nouns.
- Multiple caption templates and 2,500 caption/object pairs.
- A caption-level train/validation split: 80%/20% per noun.
- Frozen CLIP text embeddings, L2 normalization, and an embedding cache on Drive.
- Cache reload checks and a linear noun classifier sanity check.
- Validation predictions and confusion analysis.

```text
caption → frozen CLIP text encoder → normalized embedding
        → cached embedding → linear sanity classifier → noun class
```

The linear classifier is **not part of the final NOVIC architecture**. It verifies
caption-label alignment, cache integrity, the train/validation split, semantic
information in CLIP embeddings, and training pipeline correctness.

The caption CSV and selected baseline history/prediction CSVs are present locally;
the embedding cache and baseline checkpoint are stored in Drive. Both splits contain
the same nouns, so this validation does not establish unseen-noun generalization
or zero-shot image transfer.

## 8. Current Project State

```text
caption → frozen CLIP text encoder → L2-normalized embedding
        → cached embedding → linear sanity classifier → noun class
```

**This is a verified data/embedding pipeline, but it is not yet NOVIC.** The
embedding-conditioned autoregressive decoder, image transfer experiment, and
published benchmark reproduction remain pending.

## 9. Next Stage

Planned notebook: `notebooks/03_autoregressive_decoder.ipynb`.

Replace the temporary linear classifier with:

```text
CLIP embedding
  → Linear(F, P × H)
  → reshape to P prefix vectors
  → positional embeddings
  → decoder-only Transformer
  → token generation
  → EOS
```

`F` is the CLIP embedding dimension, `P` is the prefix length, and `H` is the decoder
hidden dimension. A multi-token output could be:

```text
embedding → sports → car → EOS
```

The autoregressive probability is:

$$
P(y_1, \ldots, y_T \mid z)
= \prod_{t=1}^{T} P(y_t \mid z, y_{<t}),
$$

where `z` is the conditioning CLIP embedding and the target sequence includes EOS.

## 10. Planned Decoder Components

- Target tokenizer and noun token sequences, including multi-token targets.
- BOS (beginning), EOS (end), and PAD (padding) handling.
- Embedding-to-prefix projection and prefix positional embeddings.
- Token embeddings and a decoder-only Transformer.
- Causal mask and explicit prefix-attention handling: tokens attend to the prefix
  and earlier tokens without seeing future targets.
- Teacher forcing and shifted cross-entropy loss.
- Padding masks and exclusion of PAD targets from the loss.
- Autoregressive generation with EOS stopping and a maximum generation length.
- Weight tying later, if appropriate.

Document the target tokenizer separately from the frozen CLIP text tokenizer.

## 11. Tiny-Batch Overfit Test

Before full training, the decoder must overfit **16–32 examples**. Check that loss
falls and generated sequences recover the target nouns, including multi-token
examples and EOS.

If this fails, likely bugs involve sequence shifting, causal masking, padding,
EOS handling, or prefix/token alignment. **Full training should not begin before
this test succeeds.**

## 12. Full Text-Embedding Training

After the tiny-batch test passes, train on cached text embeddings:

```text
cached CLIP text embedding → prefix mapper → Transformer decoder
                          → object noun tokens → EOS
```

This removes the temporary linear classifier from the main pipeline. Track training
loss and text-validation generation performance, save checkpoints to Drive, and
retain small result CSVs and selected plots in GitHub. This stage is pending.

## 13. Zero-Shot Text-to-Image Transfer

The key experiment substitutes image embeddings without image-label training:

```text
Training:
caption → CLIP text encoder → text embedding → decoder → noun

Inference:
image → CLIP image encoder → image embedding → same decoder → noun
```

**No image-label training will be used for the core transfer experiment.** Use the
same CLIP model, embedding dimension, and normalization convention for both paths.
Image labels can be used for evaluation. Successful text validation alone does not
establish successful image transfer; this experiment is pending.

## 14. NOVIC-Specific Refinements

### Noise

Planned ablation conditions:

- **A0:** No noise.
- **A1:** Gaussian perturbation followed by L2 normalization.
- **A2:** NOVIC-style unit-norm/angular noise following the official formulation.

The purpose is to improve robustness to the text-image modality gap. Record the
exact noise definition and strength; improved transfer is a hypothesis to test.

### Search

Start with greedy decoding, then add beam search, top-k generations, and sequence
log-probabilities. Record beam width, length scoring, and EOS behavior.

### Dataset improvement

Increase vocabulary and caption diversity gradually. Explore singular/plural
variations, multi-object descriptions, and LLM-generated captions if needed, with
explicit canonical target rules and dataset versions.

### Multiset training

Possible future M1/M2/M3-style training will use descriptions containing one, two,
or three objects. Define target ordering and handling of multiple valid nouns
before implementation. All refinements in this section are pending.

## 15. Evaluation Plan

- Exact noun-match accuracy with documented canonicalization rules.
- Top-k accuracy over distinct generated noun predictions.
- Text-validation performance and separate image-test performance.
- Qualitative failure table with targets and predictions.
- Noise ablation: A0 versus A1 versus A2.
- Greedy decoding versus beam search.
- Prefix length ablation and caption-set-size ablation.
- Vocabulary-size experiments.

Use fixed splits and comparable configurations, recording which factor changes
in each ablation. Report vocabulary and test scope alongside scores. Mini-NOVIC
results should not be presented as full paper reproduction.

## 16. Official NOVIC Reproduction

Later, use [the official NOVIC repository](https://github.com/pallgeuer/novic)
and released checkpoints to:

- Run released code/checkpoints and verify inference.
- Reproduce at least part of the reported results.
- Compare reproduced values with paper/reference values.

This is separate from our own Mini-NOVIC implementation and is **pending**. Select
a feasible benchmark and record the official code revision, checkpoint, dataset,
preprocessing, search settings, and metric. Identify any evaluated subset explicitly
rather than claiming equivalence to a full benchmark result.

## 17. Final Demonstration Plan

**Demo 1 — Standard CLIP:** show candidate-list dependence by changing the available
labels for the same image.

```text
image → CLIP → candidate label comparison → class
```

**Demo 2 — Our Mini-NOVIC:** generate a noun without a candidate list, using our
text-trained decoder. Explain the small training vocabulary and show successes
and failures.

```text
image → CLIP image embedding → our decoder → generated noun
```

**Demo 3 — Official NOVIC:** demonstrate a reproduced evaluation result and compare
it with the corresponding paper/reference value.

```text
official implementation/checkpoint → benchmark/evaluation
  → reproduced result → comparison with paper/reference value
```

The Mini-NOVIC decoder and official reproduction demonstrations depend on the
pending stages above.

## 18. Reproducibility

Experiments should record:

- Random seed and repository revision.
- CLIP model name and pretrained model identifier.
- CLIP tokenizer and target tokenizer, including special-token IDs.
- Normalization convention and embedding noise settings.
- Vocabulary version, dataset version, and split definitions.
- Embedding cache version and model/data metadata.
- Training config, optimizer settings, and decoding settings.
- Checkpoints and artifact storage locations.
- Result CSVs, selected plots, and evaluation protocol.
- Python/package versions, GPU type, and runtime settings.

Large datasets, embeddings, checkpoints, and heavy outputs live in Google Drive;
small configs and selected results live in GitHub. Record artifact paths and
checksums where possible to match checkpoints to data and configs. Drive artifacts
are not automatically available when cloning the repository.

Cloud requirements are currently not version-pinned, and local requirements are
empty. Capture the actual environment for reported runs; the current setup alone
does not guarantee identical environments or deterministic GPU results.

## 19. Current Progress Checklist

- [x] Repository structure
- [x] Git/GitHub workflow
- [x] Google Colab GPU setup
- [x] Google Drive persistent storage
- [x] CLIP exploration
- [x] Image preprocessing investigation
- [x] Text tokenization investigation
- [x] Image/text embeddings
- [x] L2 normalization
- [x] Cosine-similarity classification
- [x] Candidate-label limitation experiment
- [x] Prompt-sensitivity experiment
- [x] Embedding-space visualization
- [x] Image-text modality-gap experiment
- [x] 50-object noun vocabulary
- [x] Caption generator
- [x] Train-validation split
- [x] Frozen CLIP text embedding generation
- [x] Embedding cache
- [x] Linear sanity classifier
- [ ] Target noun tokenization
- [ ] EOS/PAD handling
- [ ] Embedding-to-prefix projection
- [ ] Autoregressive Transformer decoder
- [ ] Tiny-batch overfit test
- [ ] Full decoder training
- [ ] Greedy decoding
- [ ] Image-embedding substitution
- [ ] Zero-shot text-to-image transfer
- [ ] Embedding noise
- [ ] NOVIC-style noise
- [ ] Beam search
- [ ] Ablation studies
- [ ] Vocabulary/data scaling
- [ ] Official NOVIC benchmark reproduction
- [ ] Final demonstration

## 20. Immediate Next Step

The next notebook is **`notebooks/03_autoregressive_decoder.ipynb`**.

Its first goal is to replace the temporary linear classifier with the actual
NOVIC-style embedding-conditioned autoregressive decoder. The first gate is to
**successfully overfit a tiny batch of 16–32 examples** before full training.

## 21. Reference

**Paper:** Philipp Allgeuer, Kyra Ahrens, and Stefan Wermter.
*Unconstrained Open Vocabulary Image Classification: Zero-Shot Transfer from Text
to Image via CLIP Inversion.* WACV 2025.

- [Paper on arXiv](https://arxiv.org/abs/2407.11211)
- [Official project page](https://pallgeuer.github.io/novic/)
- [Official implementation and reproduction instructions](https://github.com/pallgeuer/novic)

This repository is an educational and experimental reproduction project focused
on understanding and demonstrating NOVIC through a staged implementation.
