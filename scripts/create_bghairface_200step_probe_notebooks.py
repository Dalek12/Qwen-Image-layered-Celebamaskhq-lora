"""Create 200-step BG/HAIR/FACE overtraining probe notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {path}")


def replace_all_cells(notebook: dict, replacements: list[tuple[str, str]]) -> None:
    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        for old, new in replacements:
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)


def write_training_notebook() -> None:
    source_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Micro_Ablation.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Training BG / HAIR / FACE 5k 200-Step Probe

Use this notebook with a GPU runtime after the BG/HAIR/FACE CPU notebook has
already created the train/val shard package. This run intentionally keeps the
same stable alpha-aware micro-ablation recipe but extends training from 100 to
200 steps, so we can test whether longer training helps or hurts the latest
BG/HAIR/FACE setup.
"""
    )["source"]
    replace_all_cells(
        notebook,
        [
            (
                "RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'",
                "RUN_NAME = 'qwen_layered_lora_bghairface_5k_200step_probe'",
            ),
            ("FULL_MAX_STEPS = 100", "FULL_MAX_STEPS = 200"),
            (
                "print('Micro ablation defaults: lr=1e-5, rank=8, alpha=8, grad_accum=4, max_steps=100, checkpoint_every=25')",
                "print('200-step probe defaults: lr=1e-5, rank=8, alpha=8, grad_accum=4, max_steps=200, checkpoint_every=25')",
            ),
            (
                "print('This alpha-weighted run uses a fresh RUN_NAME so prior micro-ablation checkpoints are not reused.')",
                "print('This 200-step probe uses a fresh RUN_NAME so prior micro-ablation checkpoints are not reused.')",
            ),
            (
                "This run trains the 5k `BG / HAIR / FACE` target scheme with LR `1e-5`, rank `8`,\nLoRA alpha `8`, 100 steps, and checkpoints every 25 steps.",
                "This run trains the 5k `BG / HAIR / FACE` target scheme with LR `1e-5`, rank `8`,\nLoRA alpha `8`, 200 steps, and checkpoints every 25 steps.",
            ),
        ],
    )
    replace_all_cells(
        notebook,
        [
            (
                """checkpoint_steps = [int(name.split('-', 1)[1]) for name in local_checkpoint_names]

if validation_rows:
    ranked_validation_rows = sorted(validation_rows, key=lambda row: row['validation_loss'])
    best_validation_row = ranked_validation_rows[0]
    best_validation_step = int(best_validation_row['step'])
    best_checkpoint_step = max(step for step in checkpoint_steps if step <= best_validation_step)
    best_checkpoint_name = f'checkpoint-{best_checkpoint_step}'
    print('Best validation step:', best_validation_step)
    print('Best validation loss:', best_validation_row['validation_loss'])
    print('Suggested checkpoint for inference:', best_checkpoint_name)
    print('Top validation checkpoints:')
    for row in ranked_validation_rows[:5]:
        ranked_step = int(row['step'])
        ranked_checkpoint_step = max(step for step in checkpoint_steps if step <= ranked_step)
        print(f"  step={ranked_step} loss={row['validation_loss']:.8f} -> checkpoint-{ranked_checkpoint_step}")
""",
                """checkpoint_steps = sorted(int(name.split('-', 1)[1]) for name in local_checkpoint_names if name.startswith('checkpoint-'))
if not checkpoint_steps:
    checkpoint_steps = sorted(
        int(path.name.split('-', 1)[1])
        for path in RUN_LOCAL.glob('checkpoint-*')
        if path.is_dir() and path.name.split('-', 1)[1].isdigit()
    )
assert checkpoint_steps, f'No checkpoint-* directories found under {RUN_LOCAL}'

def nearest_checkpoint_for_validation_step(step):
    eligible = [checkpoint_step for checkpoint_step in checkpoint_steps if checkpoint_step <= step]
    if eligible:
        return max(eligible)
    return min(checkpoint_steps, key=lambda checkpoint_step: abs(checkpoint_step - step))

if validation_rows:
    ranked_validation_rows = sorted(validation_rows, key=lambda row: row['validation_loss'])
    best_validation_row = ranked_validation_rows[0]
    best_validation_step = int(best_validation_row['step'])
    best_checkpoint_step = nearest_checkpoint_for_validation_step(best_validation_step)
    best_checkpoint_name = f'checkpoint-{best_checkpoint_step}'
    print('Best validation step:', best_validation_step)
    print('Best validation loss:', best_validation_row['validation_loss'])
    print('Suggested checkpoint for inference:', best_checkpoint_name)
    print('Available checkpoint steps:', checkpoint_steps)
    print('Top validation checkpoints:')
    for row in ranked_validation_rows[:5]:
        ranked_step = int(row['step'])
        ranked_checkpoint_step = nearest_checkpoint_for_validation_step(ranked_step)
        print(f"  step={ranked_step} loss={row['validation_loss']:.8f} -> checkpoint-{ranked_checkpoint_step}")
""",
            )
        ],
    )
    write_notebook(NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_200Step_Probe.ipynb", notebook)


CONTACT_SHEET_CELL = r"""import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CONTACT_LOCAL = METRICS_LOCAL / 'qualitative_contact_sheets'
CONTACT_LOCAL.mkdir(parents=True, exist_ok=True)

LAYER_NAMES = ['BG', 'HAIR', 'FACE']
MODEL_ORDER = ['base_model', 'checkpoint_50', 'checkpoint_200']
MODEL_LABELS = {
    'base_model': 'Base',
    'checkpoint_50': 'LoRA ckpt-50',
    'checkpoint_200': 'LoRA ckpt-200',
}

def read_rows(path):
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def metadata_path(path_str):
    return Path(str(path_str).replace('\\', '/').strip('/'))

def rgba_on_checker(path, size=(112, 112)):
    image = Image.open(path).convert('RGBA').resize(size, Image.Resampling.BICUBIC)
    tile = 14
    bg = Image.new('RGBA', size, (240, 240, 240, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=(210, 210, 210, 255))
    bg.alpha_composite(image)
    return bg.convert('RGB')

def rgb_thumb(path, size=(112, 112)):
    return Image.open(path).convert('RGB').resize(size, Image.Resampling.BICUBIC)

layer_rows = read_rows(METRICS_LOCAL / 'metrics_per_sample_layer.csv')
by_sample_variant = defaultdict(lambda: defaultdict(dict))
for row in layer_rows:
    by_sample_variant[row['sample_id_str']][row['variant_name']][row['layer_name']] = safe_float(row['iou'])

sample_scores = []
for sample_id, variants in by_sample_variant.items():
    if not all(name in variants for name in MODEL_ORDER):
        continue
    macro_by_variant = {
        variant_name: float(np.nanmean([variants[variant_name].get(layer, np.nan) for layer in LAYER_NAMES]))
        for variant_name in MODEL_ORDER
    }
    sample_scores.append({
        'sample_id': sample_id,
        'base_macro_iou': macro_by_variant['base_model'],
        'checkpoint_50_macro_iou': macro_by_variant['checkpoint_50'],
        'checkpoint_200_macro_iou': macro_by_variant['checkpoint_200'],
        'delta_200_vs_50_macro_iou': macro_by_variant['checkpoint_200'] - macro_by_variant['checkpoint_50'],
    })

sample_scores = sorted(sample_scores, key=lambda row: row['delta_200_vs_50_macro_iou'])
selected = []
if sample_scores:
    selected.extend(sample_scores[:2])
    selected.extend(sample_scores[-2:])
    selected.append(sample_scores[len(sample_scores) // 2])

deduped = {}
for row in selected:
    deduped[row['sample_id']] = row
selected = sorted(deduped.values(), key=lambda row: row['sample_id'])

thumb = (112, 112)
label_h = 28
pad = 10
cols = 1 + len(LAYER_NAMES) + len(MODEL_ORDER) * len(LAYER_NAMES)
sheet_w = pad + cols * (thumb[0] + pad)
sheet_h = pad + label_h + (thumb[1] + label_h + pad) + 34

summary = []
for score in selected:
    sample_id = score['sample_id']
    variant_dirs = {name: COMPARISON_LOCAL / name / sample_id for name in MODEL_ORDER}
    selected_sample = json.loads((variant_dirs['base_model'] / 'selected_sample.json').read_text(encoding='utf-8'))

    columns = [('Input', rgb_thumb(variant_dirs['base_model'] / 'input.png', thumb))]
    for idx, layer_name in enumerate(LAYER_NAMES):
        target_path = EVAL_LOCAL / metadata_path(selected_sample['layer_paths'][idx])
        columns.append((f'Target {layer_name}', rgba_on_checker(target_path, thumb)))
    for variant_name in MODEL_ORDER:
        model_dir = variant_dirs[variant_name]
        label_prefix = MODEL_LABELS[variant_name].replace('LoRA ', '')
        for idx, layer_name in enumerate(LAYER_NAMES):
            columns.append((f'{label_prefix} {layer_name}', rgba_on_checker(model_dir / f'layer_{idx:02d}.png', thumb)))

    sheet = Image.new('RGB', (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title = (
        f"sample {sample_id} | base={score['base_macro_iou']:.3f} | "
        f"ckpt50={score['checkpoint_50_macro_iou']:.3f} | "
        f"ckpt200={score['checkpoint_200_macro_iou']:.3f} | "
        f"200-50={score['delta_200_vs_50_macro_iou']:+.3f}"
    )
    draw.text((pad, pad), title, fill=(20, 20, 20))
    y = pad + label_h + 8
    for col_idx, (label, image) in enumerate(columns):
        x = pad + col_idx * (thumb[0] + pad)
        draw.text((x, y), label, fill=(20, 20, 20))
        sheet.paste(image, (x, y + label_h))

    out_path = CONTACT_LOCAL / f'contact_sheet_{sample_id}.png'
    sheet.save(out_path)
    summary.append({**score, 'contact_sheet': str(out_path)})

(CONTACT_LOCAL / 'contact_sheet_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

drive_contact_dir = METRICS_DRIVE / 'qualitative_contact_sheets'
if drive_contact_dir.exists():
    shutil.rmtree(drive_contact_dir)
shutil.copytree(CONTACT_LOCAL, drive_contact_dir)

print('Saved qualitative contact sheets:')
for row in summary:
    print(f" - {row['sample_id']}: 200-50 delta={row['delta_200_vs_50_macro_iou']:+.4f} -> {row['contact_sheet']}")
print('Copied contact sheets to:', drive_contact_dir)
"""


def write_eval_notebook() -> None:
    source_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Evaluate_BG_HAIR_FACE_5K_Default_Call_32.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU BG / HAIR / FACE 200-Step Probe Evaluation

Use this notebook with a GPU runtime after the 200-step probe training notebook
finishes. It compares the base model, checkpoint-50, and checkpoint-200 on the
existing 32-sample held-out BG/HAIR/FACE eval package, then writes metrics,
figures, and qualitative contact sheets.
"""
    )["source"]

    model_block_old = """MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'checkpoint_50', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': SOURCE_CHECKPOINT_NAME},
]
"""
    model_block_new = """MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'checkpoint_50', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': 'checkpoint-50'},
    {'name': 'checkpoint_200', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': 'checkpoint-200'},
]
"""

    replace_all_cells(
        notebook,
        [
            (
                """SOURCE_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'
SOURCE_RUN_DRIVE = RUNS_DRIVE_ROOT / SOURCE_RUN_NAME
SOURCE_RUN_LOCAL = Path(f'/content/{SOURCE_RUN_NAME}')
SOURCE_CHECKPOINT_NAME = 'checkpoint-50'
""",
                """BEST_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'
BEST_RUN_DRIVE = RUNS_DRIVE_ROOT / BEST_RUN_NAME
BEST_RUN_LOCAL = Path(f'/content/{BEST_RUN_NAME}')

PROBE_RUN_NAME = 'qwen_layered_lora_bghairface_5k_200step_probe'
PROBE_RUN_DRIVE = RUNS_DRIVE_ROOT / PROBE_RUN_NAME
PROBE_RUN_LOCAL = Path(f'/content/{PROBE_RUN_NAME}')

BEST_CHECKPOINT_NAME = 'checkpoint-50'
PROBE_CHECKPOINT_NAME = 'checkpoint-200'
""",
            ),
            ("COMPARISON_RUN_NAME = 'qwen_layered_lora_bghairface_5k_default_call_32_compare'", "COMPARISON_RUN_NAME = 'qwen_layered_lora_bghairface_5k_200step_probe_default_call_32_compare'"),
            ("METRICS_RUN_NAME = 'qwen_layered_lora_bghairface_5k_default_call_32_metrics'", "METRICS_RUN_NAME = 'qwen_layered_lora_bghairface_5k_200step_probe_default_call_32_metrics'"),
            (model_block_old, model_block_new),
            ("base model and `checkpoint-50`", "base model, `checkpoint-50`, and `checkpoint-200`"),
            ("The main comparison is `base_model` vs `checkpoint_50` under default-call inference.", "The main comparison is `base_model` vs `checkpoint_50` vs `checkpoint_200` under default-call inference."),
            ("print('Source run:', SOURCE_RUN_DRIVE)", "print('Best checkpoint run:', BEST_RUN_DRIVE)\nprint('200-step probe run:', PROBE_RUN_DRIVE)"),
            ("assert SOURCE_RUN_DRIVE.exists(), f'Missing source run on Drive: {SOURCE_RUN_DRIVE}'", "assert BEST_RUN_DRIVE.exists(), f'Missing best checkpoint run on Drive: {BEST_RUN_DRIVE}'\nassert PROBE_RUN_DRIVE.exists(), f'Missing 200-step probe run on Drive: {PROBE_RUN_DRIVE}'"),
            ("assert (SOURCE_RUN_DRIVE / SOURCE_CHECKPOINT_NAME).exists(), f'Missing checkpoint: {SOURCE_RUN_DRIVE / SOURCE_CHECKPOINT_NAME}'", "assert (BEST_RUN_DRIVE / BEST_CHECKPOINT_NAME).exists(), f'Missing checkpoint: {BEST_RUN_DRIVE / BEST_CHECKPOINT_NAME}'\nassert (PROBE_RUN_DRIVE / PROBE_CHECKPOINT_NAME).exists(), f'Missing checkpoint: {PROBE_RUN_DRIVE / PROBE_CHECKPOINT_NAME}'"),
            ("for path in [PROJECT_LOCAL, PACKAGE_LOCAL_ROOT, EVAL_LOCAL, SOURCE_RUN_LOCAL, COMPARISON_LOCAL, METRICS_LOCAL]:", "for path in [PROJECT_LOCAL, PACKAGE_LOCAL_ROOT, EVAL_LOCAL, BEST_RUN_LOCAL, PROBE_RUN_LOCAL, COMPARISON_LOCAL, METRICS_LOCAL]:"),
            ("shutil.copytree(SOURCE_RUN_DRIVE, SOURCE_RUN_LOCAL)", "shutil.copytree(BEST_RUN_DRIVE, BEST_RUN_LOCAL)\nshutil.copytree(PROBE_RUN_DRIVE, PROBE_RUN_LOCAL)"),
            ("print('Copied repo, source run, and extracted 32-sample eval package.')", "print('Copied repo, best/probe runs, and extracted 32-sample eval package.')"),
            ("model_variant.get('run_dir', SOURCE_RUN_LOCAL) / model_variant['checkpoint_name']", "model_variant['run_dir'] / model_variant['checkpoint_name']"),
        ],
    )

    model_block_probe = """MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'checkpoint_50', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': 'checkpoint-50'},
    {'name': 'checkpoint_200', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': 'checkpoint-200'},
]
"""
    model_block_probe_fixed = """MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'checkpoint_50', 'type': 'lora', 'run_dir': BEST_RUN_LOCAL, 'checkpoint_name': BEST_CHECKPOINT_NAME},
    {'name': 'checkpoint_200', 'type': 'lora', 'run_dir': PROBE_RUN_LOCAL, 'checkpoint_name': PROBE_CHECKPOINT_NAME},
]
"""
    replace_all_cells(notebook, [(model_block_probe, model_block_probe_fixed)])

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "CONTACT_LOCAL = METRICS_LOCAL / 'qualitative_contact_sheets'" in "".join(cell.get("source", [])):
            cell["source"] = CONTACT_SHEET_CELL.splitlines(keepends=True)

    write_notebook(NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Evaluate_BG_HAIR_FACE_5K_200Step_Probe_Default_Call_32.ipynb", notebook)


def main() -> None:
    write_training_notebook()
    write_eval_notebook()


if __name__ == "__main__":
    main()
