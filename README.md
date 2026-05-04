# Qwen-Image-Layered Portrait Layer Decomposition

This project adapts `Qwen/Qwen-Image-Layered` to portrait layer decomposition
using CelebAMask-HQ. The main experiment trains and evaluates a LoRA for a
3-layer target representation:

- `BG`
- `HAIR`
- `FACE`

where `FACE = FACE_SKIN + EYES + MOUTH`.

## Current Result

The infrastructure works end to end: preprocessing, metadata recovery, train/val
shard packaging, GPU LoRA training, checkpoint comparison, and numeric evaluation.
The model-quality result is more modest. On a 32-sample held-out evaluation, the
best LoRA checkpoint slightly improves target-mask alignment over the base model,
mostly on the `FACE` layer, but it does not solve layer separation.

Summary from `results/qwen_layered_lora_bghairface_5k_default_call_32_metrics`:

| model | macro IoU | soft alpha IoU | alpha MAE | BG IoU | HAIR IoU | FACE IoU |
|---|---:|---:|---:|---:|---:|---:|
| base model | 0.2461 | 0.2469 | 0.5461 | 0.3229 | 0.1607 | 0.2547 |
| checkpoint-50 LoRA | 0.2547 | 0.2554 | 0.5425 | 0.3229 | 0.1610 | 0.2800 |

The final report also includes a 200-step BG/HAIR/FACE probe. Those outputs are
stored in `results/qwen_layered_lora_bghairface_5k_200step_probe_default_call_32_metrics`.
The 200-step checkpoint is stable, but the improvement is still small and layer
overlap remains high:

| model | macro IoU | soft alpha IoU | alpha MAE | BG IoU | HAIR IoU | FACE IoU |
|---|---:|---:|---:|---:|---:|---:|
| base model | 0.2461 | 0.2469 | 0.5461 | 0.3229 | 0.1607 | 0.2547 |
| checkpoint-50 LoRA | 0.2547 | 0.2554 | 0.5425 | 0.3229 | 0.1610 | 0.2800 |
| checkpoint-200 LoRA | 0.2585 | 0.2592 | 0.5512 | 0.3229 | 0.1578 | 0.2947 |

Layer-health diagnostics still show high overlap and frequent full-canvas layers,
so the conclusion is that target representation alone was not sufficient. Future
work should focus on explicit alpha/separation losses, segmentation-guided cleanup,
or more model-aligned target layouts.

## Repository Contents

- `preprocess_celebmaskhq/`: CelebAMask-HQ preprocessing and layered export code.
- `train_celebmaskhq/`: dataset loaders used by training.
- `scripts/`: preprocessing, packaging, training, comparison, evaluation, and
  notebook-generation scripts.
- `notebooks/`: selected Colab notebooks for the reproducible BG/HAIR/FACE runs.
- `results/`: small numeric metric outputs and aggregate plots.
- `FUTURE_LAYER_SEPARATION_APPROACHES.md`: future-work notes.
- `final_blog.pdf` and `blog_report_overleaf.tex`: final report artifact and
  Overleaf source.

## Data Policy

Raw CelebAMask-HQ images, processed layered datasets, tar shards, checkpoints, and
full visual comparison folders are intentionally not included in this repository.
They are large and contain dataset-derived face images. The notebooks regenerate
the required processed artifacts from an external CelebAMask-HQ download.

Expected Google Drive layout in Colab:

```text
/content/drive/MyDrive/CelebMaskHQ_Colab/
  repo/
  raw/
    CelebA-HQ-img/
    CelebAMask-HQ-mask-anno/
  processed/
  runs/
```

## Reproduction Order

1. Copy this repository to:

   ```text
   /content/drive/MyDrive/CelebMaskHQ_Colab/repo
   ```

2. Place the raw CelebAMask-HQ files under:

   ```text
   /content/drive/MyDrive/CelebMaskHQ_Colab/raw
   ```

3. Run the CPU preprocessing/package notebook:

   ```text
   notebooks/CelebAMaskHQ_CPU_BG_HAIR_FACE_5K_Preprocess_And_Package.ipynb
   ```

4. Run the main GPU training notebook:

   ```text
   notebooks/CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Micro_Ablation.ipynb
   ```

5. Run the checkpoint comparison notebook:

   ```text
   notebooks/CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Micro_Ablation_Checkpoints.ipynb
   ```

6. Run the 32-sample evaluation package notebook:

   ```text
   notebooks/CelebAMaskHQ_CPU_BG_HAIR_FACE_5K_Package_32_Sample_Eval.ipynb
   ```

7. Run the 32-sample default-call metric notebook:

   ```text
   notebooks/CelebAMaskHQ_GPU_Evaluate_BG_HAIR_FACE_5K_Default_Call_32.ipynb
   ```

8. To reproduce the 200-step probe used in the final report, run:

   ```text
   notebooks/CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_200Step_Probe.ipynb
   notebooks/CelebAMaskHQ_GPU_Evaluate_BG_HAIR_FACE_5K_200Step_Probe_Default_Call_32.ipynb
   ```

The numeric CSV/JSON outputs and aggregate plots used in the report are included
under `results/`. The report's qualitative contact sheets are included only for
the two samples referenced by the LaTeX source.

## Notes

The included numeric metrics should be interpreted as alignment to the
CelebAMask-derived `BG / HAIR / FACE` target representation, not as a complete
measure of human-perceived editability.
