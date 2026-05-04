# Experiment Timeline

This file is a chronological record of the major experiments, infrastructure changes, and conclusions in the project.

Workspace:

- `D:\testing123\CelebAMask-HQ\CelebAMask-HQ`

Related handoff files:

- `D:\testing123\CelebAMask-HQ\CelebAMask-HQ\SESSION_HANDOFF_GPT55.md`
- `D:\testing123\CelebAMask-HQ\CelebAMask-HQ\NEXT_SESSION_PROMPT_GPT55.md`

## 1. Initial Goal

Train a LoRA on top of `Qwen/Qwen-Image-Layered` so that a portrait image can be decomposed into editable RGBA portrait layers using `CelebAMask-HQ` as supervision.

Initial intended target representation:

- `BG`
- `HAIR`
- `FACE_SKIN`
- `EYES`
- `MOUTH`
- optionally additional canonical slots in preprocessing:
  - `CLOTH`
  - `ACCESSORY`

## 2. Starting Point

What existed at the beginning:

- a local codebase for `CelebAMask-HQ`
- older preprocessing assumptions based on a more fixed-slot representation
- partial training/inference logic
- no stable end-to-end Colab pipeline

Important early reality:

- the raw dataset copied to Google Drive was not perfectly complete
- some image IDs and mask IDs did not line up

## 3. Early Infrastructure Phase

### 3.1 Preprocessing redesign

Main change:

- preprocessing was extended to produce a generic layered export suitable for `Qwen-Image-Layered`

Processed dataset structure became:

- `layered_composites/`
- `layered_layers/<sample_id>/`
- `metadata/layered_samples.jsonl`

Why this mattered:

- training needed a generic per-sample layered representation rather than the earlier fixed training contract

### 3.2 Handling incomplete raw data

Problem:

- preprocessing would fail because some IDs had an image but no masks, or masks but no image

Fix:

- preprocessing was changed to use the intersection of valid image IDs and valid mask IDs
- missing IDs were recorded instead of causing hard failure

Result:

- the pipeline became tolerant enough to process the usable majority of the raw dataset

## 4. Colab Workflow Stabilization

### 4.1 CPU/GPU separation

Decision:

- preprocessing and packaging were handled in CPU-oriented notebooks
- training and evaluation were handled in GPU notebooks

Reason:

- raw dataset was too large and too slow to keep reprocessing directly inside short GPU sessions

### 4.2 Drive-first storage strategy

Decision:

- raw and processed datasets stayed on Drive
- smaller packaged artifacts were copied to local `/content` for faster GPU-side training

Reason:

- Drive-backed random I/O was too slow and fragile for direct GPU training over many tiny files

### 4.3 Stale Colab copy issue

Problem:

- updating files in the Drive repo did not automatically update `/content/project`

Fix:

- freshness-check cells were added
- scripts were checked for marker strings
- Drive repo sync + repo-copy became an explicit workflow step

Result:

- reduced false debugging caused by stale runtime files

## 5. Metadata Recovery and Packaging Phase

### 5.1 Full preprocessing became too slow

Problem:

- full preprocessing took too long for a single Colab session
- interrupted runs left many processed artifacts but sometimes no usable metadata

Fix:

- `scripts/recover_processed_celebmaskhq_metadata.py` was added
- it rebuilt metadata from existing processed artifacts
- later it was updated to write JSONL incrementally and show progress

Result:

- partially processed datasets became usable for training

### 5.2 Giant tar bottleneck

Problem:

- creating one huge tar archive for all processed outputs was slow and fragile
- interruptions could waste hours of work

Fix:

- train/val shard packaging was introduced:
  - `scripts/package_qwen_layered_trainval_shards.py`
- train and val only were packaged
- test was kept separate
- shard reuse / resume behavior was added

Result:

- packaging became much more resilient
- only the data needed for training was transferred

### 5.3 Test evaluation packaging

Need:

- compare checkpoints on small held-out test subsets cheaply

Fix:

- `scripts/package_qwen_layered_eval_subset.py`
- small test-only package
- dedicated compare notebooks

Result:

- checkpoint evaluation became much cheaper and more repeatable

## 6. Trainer Bring-Up Phase

### 6.1 API mismatches and runtime breakages

Problem:

- the trainer did not initially match current `diffusers`, Qwen layered pipeline, and PEFT behavior

Types of issues:

- import path issues
- latent packing/unpacking mismatches
- transformer calling-contract mismatches
- save-path incompatibilities
- prompt-encoding graph reuse issues

Fix:

- `scripts/train_qwen_image_layered_lora.py` was repeatedly updated
- generic layered dataset integration was added
- latent packing/unpacking behavior was aligned with the real model contract
- save logic and prompt encoding were stabilized

Result:

- smoke training eventually ran end to end

### 6.2 Checkpoint resume issues

Problem:

- resume-from-checkpoint initially failed or resumed with frozen LoRA adapters
- some runtime combinations produced empty optimizer parameter lists

Fix:

- checkpoint save logic was expanded to include:
  - optimizer state
  - LR scheduler state
  - training state
- resume logic reloaded those states
- explicit adapter reactivation logic was added after resume
- compatibility logic was added for multiple PEFT/runtime variants

Result:

- resumed training from mirrored checkpoints became reliable enough for repeated Colab use

## 7. First Meaningful Dataset Scale-Up

Recovered dataset size around the main scale-up period:

- train: `15,672`
- val: `1,943`
- test: `1,926`
- train + val: `17,615`

This was enough to justify full train/val experimentation.

## 8. Full Train/Val Shard Run

Notebook:

- `notebooks/CelebAMaskHQ_GPU_Training_TrainVal_Shards.ipynb`

Representative settings:

- `learning_rate = 1e-4`
- `rank = 16`
- `lora_alpha = 16`
- `gradient_accumulation_steps = 8`
- `max_train_steps = 2000`
- checkpoint every `200`

Observed training behavior:

- loss curve looked numerically plausible
- validation best checkpoint was around `checkpoint-1200`

Held-out visual comparison:

- outputs on held-out test samples were badly degraded
- many layers became colorful structured noise

Important manual comparison sample:

- sample `10832`

Conclusion:

- first large run trained numerically but failed visually
- this was a real model-quality failure, not just an inference notebook issue

## 9. Base vs Smoke vs Full-Run Comparison

Purpose:

- determine whether the full-run failure came from the inference pipeline or the trained LoRA itself

Compared variants:

- base `Qwen-Image-Layered`
- smoke LoRA
- full-run checkpoints

Observation:

- base model produced structured, usable outputs
- smoke LoRA still preserved some structure
- full-run checkpoints were much noisier and often unusable

Conclusion:

- the inference pipeline was not the main problem
- the aggressive full fine-tuning regime was damaging the model

## 10. Stability Rerun

Notebook:

- `notebooks/CelebAMaskHQ_GPU_Training_TrainVal_Shards_Stability_Rerun.ipynb`

Representative settings:

- `learning_rate = 5e-5`
- `rank = 16`
- `lora_alpha = 16`
- `gradient_accumulation_steps = 4`
- `max_train_steps = 800`
- checkpoint every `100`

Observed results:

- best validation checkpoint appeared much earlier, around `checkpoint-200`
- later checkpoints generally worsened
- the run looked more stable numerically than the full 2000-step run

Visual outcome:

- still not clearly better than base
- noise collapse was less severe, but results were still poor enough to reject

Conclusion:

- overtraining was part of the problem
- but reducing step count and LR alone did not solve the deeper issue

## 11. First Micro-Ablation Run

Notebook:

- `notebooks/CelebAMaskHQ_GPU_Training_TrainVal_Shards_Micro_Ablation.ipynb`

Representative settings:

- `learning_rate = 1e-5`
- `rank = 8`
- `lora_alpha = 8`
- `gradient_accumulation_steps = 4`
- `max_train_steps = 100`
- checkpoint every `25`

Observed training results:

- best validation checkpoint was around `checkpoint-50`
- training was much less explosive

Visual outcome:

- outputs were stable
- but they looked very close to the base model
- little or no meaningful improvement was visible

Conclusion:

- very weak LoRA settings avoided collapse
- but the adaptation was almost a no-op

## 12. Processed Target Audit

Need:

- determine whether the supervision targets themselves were broken

Tooling:

- `scripts/audit_processed_layered_samples.py`
- `notebooks/CelebAMaskHQ_CPU_Audit_Processed_Layers.ipynb`

Manual audit result:

- sampled target layers looked semantically correct
- example manually reviewed:
  - sample `07507`
  - layers looked reasonable for:
    - `BG`
    - `HAIR`
    - `FACE_SKIN`
    - `EYES`
    - `MOUTH`

Conclusion:

- target layers were not obviously nonsensical
- the problem likely was not trivial label corruption

## 13. Transparency / Loss Mismatch Hypothesis

Hypothesis:

- sparse RGBA layers had large transparent regions
- transparent pixels might still carry hidden RGB values
- plain latent MSE might let transparent canvas dominate learning signal

This could explain:

- why stronger runs drifted into noise
- why sparse facial layers were especially problematic

## 14. Alpha-Aware Training Fix

Code changes:

- `train_celebmaskhq/generic_layered_dataset.py`
  - target RGB premultiplied by alpha
- `scripts/train_qwen_image_layered_lora.py`
  - latent alpha-weight map built from target alpha channel
  - squared error weighted by alpha-derived visibility weight

Interpretation:

- visible content matters more
- transparent background matters less

## 15. Alpha-Weighted Micro-Ablation Run

Run name:

- `qwen_layered_lora_trainval_micro_ablation_alpha_weighted`

Representative settings:

- same weak micro-ablation settings as before
- alpha-aware loader + alpha-weighted loss now active

Observed numeric results:

- best validation checkpoint again around `checkpoint-50`
- validation loss improved substantially relative to the earlier micro-ablation
- final train loss and validation loss were much lower than the earlier weak run

Visual outcome:

- catastrophic noise collapse was avoided
- outputs remained very close to the base model
- only minor differences were visible
- no strong practical improvement over baseline

Conclusion:

- alpha-aware training fixed an important stability issue
- but it still did not create convincing visual gains

## 16. Main Final Modeling Interpretation Before Blog Stage

At this point, the project had strong evidence for the following:

### What seems solved

- pipeline reliability
- preprocessing viability
- packaging and resume workflow
- transparency-related collapse mitigation

### What remains unsolved

- how to actually improve decomposition quality beyond the base model

### Most likely explanation

- the current 5-layer target representation is not well aligned with the base model’s native decomposition behavior

In other words:

- the model may prefer broader editable portrait chunks
- the current target layers may be too segmentation-like or too fine-grained

## 17. Recommended Next Experiment After This Timeline

The recommended next step was:

- stop stretching the same 5-layer scheme to 200-500 steps just to see if it improves
- instead, change the target representation

Recommended first new target scheme:

- `BG`
- `HAIR`
- `FACE`

where:

- `FACE = FACE_SKIN + EYES + MOUTH`

Recommended first training recipe for that new scheme:

- keep the stable weak alpha-aware micro-ablation settings:
  - `learning_rate = 1e-5`
  - `rank = 8`
  - `lora_alpha = 8`
  - `gradient_accumulation_steps = 4`
  - `max_train_steps = 100`
  - checkpoint every `25`

Reason:

- isolate target-representation change first
- avoid mixing it with aggressive hyperparameter changes

## 18. Blog-Stage Framing

By the blog stage, the project was best framed as:

- an engineering + experimentation write-up

not yet as:

- an academic paper

Reason:

- the strongest contribution was the robust experimental pipeline and the diagnostic findings
- there was not yet a clear model-quality win over the base model

Suggested blog framing:

- building a layered portrait fine-tuning pipeline for `Qwen-Image-Layered`
- what worked in the engineering stack
- what failed in the modeling setup
- what target alignment taught us

## 19. Short Timeline Summary

If the next session only needs the shortest version:

1. Built a generic layered preprocessing pipeline from CelebAMask-HQ.
2. Stabilized Colab workflows, packaging, checkpointing, and metadata recovery.
3. Ran a large train/val LoRA fine-tuning experiment.
4. Found that stronger training produced noisy outputs.
5. Weakened training and later added alpha-aware fixes.
6. Prevented catastrophic collapse, but still got little visual improvement over the base model.
7. Concluded that the likely next breakthrough depends on changing the target representation, not just tuning the same 5-layer recipe longer.
