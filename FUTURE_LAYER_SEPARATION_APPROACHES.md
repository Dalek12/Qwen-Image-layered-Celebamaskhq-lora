# Future Approaches For Improving Portrait Layer Separation

This project found that the current LoRA setup can slightly improve some numeric
alignment metrics, especially on the `FACE` layer, but it does not reliably solve
portrait-level layer separation. The main remaining issue is structural: generated
layers often overlap heavily or become nearly full-canvas layers.

These are the most promising next directions after the current blog/report stage.

## 1. Add Explicit Alpha And Separation Losses

The most direct next step is to modify training so it supervises the layer alpha
structure more explicitly, not only the generated layered image representation.

Possible additions:

- alpha mask BCE or Dice loss against the target `BG / HAIR / FACE` alpha masks
- overlap penalty between predicted layer alpha masks
- full-canvas penalty for non-background layers
- optional sparsity or coverage regularization per layer

This would target the exact failure mode seen in the metrics: high inter-layer
overlap and frequent full-canvas predicted layers. It is likely the strongest
research direction, but it requires deeper changes to the training loop.

## 2. Hybrid Qwen + Segmentation-Guided Cleanup

Another practical path is to keep Qwen-Image-Layered as the generative layer
source, then use a face parsing or segmentation prior to clean the alpha channels.

For this project, the cleanup masks could follow the same portrait grouping:

- `BG`: background/remainder mask
- `HAIR`: hair mask
- `FACE`: skin + eyes + mouth mask

This would likely improve numeric IoU and visual separation quickly. The tradeoff
is that the result becomes a hybrid system rather than a pure LoRA adaptation of
Qwen-Image-Layered.

## 3. Try A More Model-Aligned Target Representation

The strict `BG / HAIR / FACE` target may still be too segmentation-like for the
base model's natural decomposition behavior. A future experiment could use broader
or more compositional targets, such as:

- `BG / FULL_SUBJECT / HAIR_DETAIL`
- `BG / HEAD_AND_HAIR / FACE_FEATURES`
- `BG / SUBJECT / FACE`

The goal would be to match how the pretrained layered model already tends to
organize images, instead of forcing it into a semantic segmentation layout.

## 4. Curate A Smaller, Cleaner Training Set

The current 5k subset is broad, but CelebAMask-HQ includes difficult cases such as
occlusions, accessories, ambiguous hair boundaries, unusual poses, and inconsistent
face/neck regions. A curated subset of cleaner frontal portraits may provide a
stronger supervision signal.

Possible filters:

- frontal or near-frontal faces
- visible hair and face
- minimal occlusion from hands, hats, microphones, or accessories
- balanced hair coverage
- fewer extreme crops

This may help determine whether the bottleneck is noisy supervision or model/target
misalignment.

## Suggested Blog Wording

The future-work section can mention these approaches, but should clearly separate
them from the completed experiment results. A concise version:

> The evaluation suggests that target representation alone was not enough to obtain
> reliable editable portrait layers. The LoRA produced a small improvement in
> FACE-layer alignment, but the generated layers still showed high overlap and
> frequent full-canvas alpha masks. Future work should therefore focus on training
> objectives that directly supervise alpha separation, such as mask Dice/BCE losses,
> inter-layer overlap penalties, or full-canvas penalties for non-background layers.
> A second practical direction is a hybrid pipeline that combines Qwen-Image-Layered
> generation with segmentation-guided alpha cleanup. Finally, a more model-aligned
> target representation or a curated clean-portrait subset may help determine
> whether the remaining limitation comes from noisy supervision or from a mismatch
> between semantic segmentation targets and the pretrained layered model's natural
> decomposition behavior.

