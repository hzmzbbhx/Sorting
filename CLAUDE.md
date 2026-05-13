# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AnomalyCLIP is a zero-shot anomaly detection (ZSAD) framework (ICLR 2024) that adapts CLIP (ViT-L/14@336px) for detecting anomalies across diverse domains without target-dataset training samples. The core idea: learn **object-agnostic** text prompts that capture generic normality/abnormality rather than foreground-object semantics, enabling cross-domain generalization.

## Commands

### Generate dataset JSON (required before any run)
```bash
cd generate_dataset_json
python mvtec.py       # MVTec AD (multi-class anomalies)
python visa.py         # VisA
python SDD.py          # SDD / single-class datasets
# ...one script per dataset — match your dataset folder structure
```

### Quick test with pre-trained weights
```bash
bash test.sh
```
Or directly:
```bash
python test.py --dataset mvtec --data_path ./data/mvtec --save_path ./results/zero_shot \
  --checkpoint_path ./checkpoints/9_12_4_multiscale/epoch_15.pth \
  --features_list 6 12 18 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4
```

### Train
```bash
bash train.sh
```
Or directly:
```bash
python train.py --dataset visa --train_data_path ./data/visa --save_path ./checkpoints/ \
  --features_list 6 12 18 24 --image_size 518 --batch_size 8 --epoch 15 \
  --depth 9 --n_ctx 12 --t_n_ctx 4
```

### Key test.py arguments
- `--metrics`: `image-level`, `pixel-level`, or `image-pixel-level` (default)
- `--features_list`: which ViT layers to extract patch features from (default `6 12 18 24`)
- `--feature_map_layer`: which indices in `features_list` to use for anomaly maps (default `0 1 2 3` = all)
- `--sigma`: Gaussian filter sigma for anomaly map smoothing (default 4)

## Architecture

### DPAM (Dual-Path Attention Mechanism)
The vision encoder is modified to create two parallel information paths. `VisionTransformer.DAPM_replace(DPAM_layer=20)` replaces the top 20 transformer layers' self-attention with a custom `Attention` module that computes both original self-attention and a v-v self-attention variant (where k and q are replaced by v). During inference, patch tokens from the "new" path are used for anomaly map generation.

### PromptLearner (`prompt_ensemble.py`)
`AnomalyCLIP_PromptLearner` learns object-agnostic text embeddings:
- **Learnable context vectors** (`ctx_pos`, `ctx_neg`): inserted into normal/abnormal prompt templates surrounding the word "object"
- **Compound prompts** (`compound_prompts_text`): deep learnable tokens inserted into intermediate text encoder layers (controlled by `--depth` / `learnabel_text_embedding_depth`)
- Forward returns: `(prompts, tokenized_prompts, compound_prompts_text)` consumed by `model.encode_text_learn()`

### Inference flow (`test.py`)
1. Load frozen ViT-L/14@336px + apply DAPM_replace
2. Load trained PromptLearner weights
3. Compute text features for normal/abnormal prompts → stack into `[normal, abnormal]` pairs
4. For each image: extract multi-scale patch features, compute cosine similarity with text features, reshape to spatial similarity maps, average across layers, apply Gaussian smoothing → anomaly map
5. Image-level score: global image feature @ text features → softmax → abnormal probability

### Training flow (`train.py`)
- Only PromptLearner parameters are trained (ViT backbone is frozen)
- Loss = `λ * (FocalLoss + DiceLoss on patch similarity maps)` + cross-entropy on image-level predictions
- `λ = 4` by default

### Dataset format (`dataset.py`)
Each dataset folder must contain a `meta.json` with structure:
```json
{"test": {"class_name": [{"img_path": "...", "mask_path": "...", "cls_name": "...", "specie_name": "...", "anomaly": 0/1}, ...]}}
```
Scripts in `generate_dataset_json/` produce this JSON. To add a custom dataset: (1) create a generator script, (2) add the dataset name + class list to `generate_class_info()` in `dataset.py`.

### Key model components (`AnomalyCLIP_lib/`)
| File | Purpose |
|---|---|
| `AnomalyCLIP.py` | `AnomalyCLIP` model, `VisionTransformer`, `Transformer`, `Attention` (DPAM), `ResidualAttentionBlock`, `ResidualAttentionBlock_learnable_token` |
| `model_load.py` | `load()`, `compute_similarity()`, `get_similarity_map()` |
| `build_model.py` | Instantiates `AnomalyCLIP` vs. original `CLIP` based on whether `design_details` is provided |
| `CLIP.py` | Original CLIP model (used when no DPAM/prompt learning) |
| `constants.py` | `OPENAI_DATASET_MEAN`, `OPENAI_DATASET_STD` |

### Pre-trained weights
The model downloads `ViT-L-14-336px.pt` from OpenAI. The cache path is hardcoded to `/remote-home/iot_zhouqihang/root/.cache/clip` in `model_load.py:38` — update this for other machines.

## Environment
- PyTorch 2.0.0+
- Single NVIDIA RTX 3090 24GB (as reported in paper)
- Batch size 8, 15 epochs, Adam lr=0.001, betas=(0.5, 0.999)
