"""Create 32-sample numeric evaluation notebooks for BG/HAIR/FACE."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {path}")


def write_cpu_eval32_package_notebook() -> None:
    cells = [
        markdown(
            """# CelebAMask-HQ CPU BG / HAIR / FACE 32-Sample Eval Package

Use this notebook with a CPU runtime. It creates a 32-sample held-out test eval
package from the already processed 5k `BG / HAIR / FACE` dataset.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            """from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/CelebMaskHQ_Colab')
REPO_DRIVE = DRIVE_ROOT / 'repo'
PROCESSED_DRIVE_ROOT = DRIVE_ROOT / 'processed'
PROJECT_LOCAL = Path('/content/project')

PROCESSED_NAME = 'processed_celebmaskhq_bghairface_5k'
PROCESSED_DRIVE = PROCESSED_DRIVE_ROOT / PROCESSED_NAME

EVAL32_PACKAGE_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval_32'
EVAL32_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval_32_package'
EVAL32_OUTPUT_DRIVE = PROCESSED_DRIVE_ROOT / EVAL32_OUTPUT_NAME

MAX_EVAL_SAMPLES = 32
FORCE_REBUILD_EVAL32_PACKAGE = False

print('Repo on Drive:', REPO_DRIVE)
print('Processed dataset:', PROCESSED_DRIVE)
print('32-sample eval package:', EVAL32_OUTPUT_DRIVE)
"""
        ),
        code(
            """import shutil

assert REPO_DRIVE.exists(), f'Missing repo folder on Drive: {REPO_DRIVE}'
assert PROCESSED_DRIVE.exists(), f'Missing processed BG/HAIR/FACE 5k dataset: {PROCESSED_DRIVE}'
assert (PROCESSED_DRIVE / 'metadata' / 'layered_samples.jsonl').exists(), 'Missing processed layered metadata.'

if PROJECT_LOCAL.exists():
    shutil.rmtree(PROJECT_LOCAL)
shutil.copytree(REPO_DRIVE, PROJECT_LOCAL)
print('Copied repo to', PROJECT_LOCAL)
"""
        ),
        code(
            """import json

mapping = json.loads((PROCESSED_DRIVE / 'metadata' / 'mapping.json').read_text(encoding='utf-8'))
rows = [
    json.loads(line)
    for line in (PROCESSED_DRIVE / 'metadata' / 'layered_samples.jsonl').read_text(encoding='utf-8').splitlines()
    if line.strip()
]
test_rows = [row for row in rows if row.get('split') == 'test']

assert mapping['layered_export']['scheme'] == 'bg_hair_face'
assert mapping['layered_export']['target_layer_names'] == ['BG', 'HAIR', 'FACE']
assert len(test_rows) >= MAX_EVAL_SAMPLES, f'Need at least {MAX_EVAL_SAMPLES} test rows; found {len(test_rows)}'
assert all(row['layer_names'] == ['BG', 'HAIR', 'FACE'] for row in test_rows), test_rows[0]
assert all(row['layer_count'] == 3 for row in test_rows), test_rows[0]

print('Processed dataset verification passed.')
print('Available test rows:', len(test_rows))
"""
        ),
        code(
            """import json
import shutil
import subprocess
import sys

if FORCE_REBUILD_EVAL32_PACKAGE and EVAL32_OUTPUT_DRIVE.exists():
    shutil.rmtree(EVAL32_OUTPUT_DRIVE)

manifest_path = EVAL32_OUTPUT_DRIVE / 'package_manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['target_split'] == 'test', manifest['target_split']
    assert manifest['sample_count'] == MAX_EVAL_SAMPLES, manifest['sample_count']
    assert manifest['package_name'] == EVAL32_PACKAGE_NAME, manifest['package_name']
    print('32-sample eval package already exists and matches this notebook; skipping rebuild.')
else:
    command = [
        sys.executable,
        '-u',
        'scripts/package_qwen_layered_eval_subset.py',
        '--processed-root',
        str(PROCESSED_DRIVE),
        '--output-root',
        str(EVAL32_OUTPUT_DRIVE),
        '--package-name',
        EVAL32_PACKAGE_NAME,
        '--split',
        'test',
        '--max-samples',
        str(MAX_EVAL_SAMPLES),
        '--sample-strategy',
        'spaced',
        '--progress-every',
        '1',
    ]
    print('Running command:', ' '.join(command))
    subprocess.run(command, cwd=PROJECT_LOCAL, check=True)

assert manifest_path.exists(), f'Missing eval package manifest: {manifest_path}'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
print('32-sample eval package ready at:', EVAL32_OUTPUT_DRIVE)
print('Selected test sample ids:', manifest['selected_sample_ids'])
"""
        ),
        markdown(
            """## Result

The 32-sample eval package is ready on Drive:

- `processed/processed_celebmaskhq_bghairface_5k_test_eval_32_package`
"""
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "CelebAMaskHQ_CPU_BG_HAIR_FACE_5K_Package_32_Sample_Eval.ipynb", cells)


def write_gpu_eval32_notebook() -> None:
    cells = [
        markdown(
            """# CelebAMask-HQ GPU BG / HAIR / FACE Default-Call 32-Sample Evaluation

Use this notebook with a GPU runtime. It runs the base model and `checkpoint-50`
with the default/no-explicit-prompt inference path on a 32-sample held-out test
package, then computes numeric metrics on CPU.
"""
        ),
        code(
            """from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        code(
            """from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/CelebMaskHQ_Colab')
REPO_DRIVE = DRIVE_ROOT / 'repo'
PROCESSED_DRIVE_ROOT = DRIVE_ROOT / 'processed'
RUNS_DRIVE_ROOT = DRIVE_ROOT / 'runs'

PROJECT_LOCAL = Path('/content/project')
PACKAGE_LOCAL_ROOT = Path('/content/test_eval_package_default_call_32')

EVAL_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval_32_package'
EVAL_OUTPUT_DRIVE = PROCESSED_DRIVE_ROOT / EVAL_OUTPUT_NAME
PACKAGE_MANIFEST_DRIVE = EVAL_OUTPUT_DRIVE / 'package_manifest.json'

SOURCE_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'
SOURCE_RUN_DRIVE = RUNS_DRIVE_ROOT / SOURCE_RUN_NAME
SOURCE_RUN_LOCAL = Path(f'/content/{SOURCE_RUN_NAME}')
SOURCE_CHECKPOINT_NAME = 'checkpoint-50'

COMPARISON_RUN_NAME = 'qwen_layered_lora_bghairface_5k_default_call_32_compare'
COMPARISON_LOCAL = Path(f'/content/{COMPARISON_RUN_NAME}')
COMPARISON_DRIVE = RUNS_DRIVE_ROOT / COMPARISON_RUN_NAME

METRICS_RUN_NAME = 'qwen_layered_lora_bghairface_5k_default_call_32_metrics'
METRICS_LOCAL = Path(f'/content/{METRICS_RUN_NAME}')
METRICS_DRIVE = RUNS_DRIVE_ROOT / METRICS_RUN_NAME

MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'checkpoint_50', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': SOURCE_CHECKPOINT_NAME},
]

MAX_COMPARE_SAMPLES = 32
MIXED_PRECISION = 'bf16'
RESOLUTION = 640
MAX_LAYERS = 3
INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 1337
FORCE_REBUILD_COMPARISON = True
FORCE_REBUILD_METRICS = True

print('Repo on Drive:', REPO_DRIVE)
print('Eval package:', EVAL_OUTPUT_DRIVE)
print('Source run:', SOURCE_RUN_DRIVE)
print('Comparison output:', COMPARISON_DRIVE)
print('Metrics output:', METRICS_DRIVE)
"""
        ),
        code(
            """!nvidia-smi
"""
        ),
        code(
            """!pip uninstall -y torchao
!pip install -U accelerate peft bitsandbytes sentencepiece protobuf
!pip install -U "transformers>=4.51.3"
!pip install git+https://github.com/huggingface/diffusers.git
"""
        ),
        code(
            """import json
import shutil
import subprocess

assert REPO_DRIVE.exists(), f'Missing repo folder on Drive: {REPO_DRIVE}'
assert PACKAGE_MANIFEST_DRIVE.exists(), f'Missing package manifest on Drive: {PACKAGE_MANIFEST_DRIVE}'
assert SOURCE_RUN_DRIVE.exists(), f'Missing source run on Drive: {SOURCE_RUN_DRIVE}'
assert (SOURCE_RUN_DRIVE / SOURCE_CHECKPOINT_NAME).exists(), f'Missing checkpoint: {SOURCE_RUN_DRIVE / SOURCE_CHECKPOINT_NAME}'

package_manifest = json.loads(PACKAGE_MANIFEST_DRIVE.read_text(encoding='utf-8'))
assert package_manifest['target_split'] == 'test', package_manifest['target_split']
assert package_manifest['sample_count'] == MAX_COMPARE_SAMPLES, package_manifest['sample_count']

PACKAGE_NAME = package_manifest['package_name']
METADATA_TAR_DRIVE = EVAL_OUTPUT_DRIVE / package_manifest['metadata_tar']
DATA_TAR_DRIVE = EVAL_OUTPUT_DRIVE / package_manifest['data_tar']
EVAL_LOCAL = Path('/content') / PACKAGE_NAME
METADATA_TAR_LOCAL = PACKAGE_LOCAL_ROOT / package_manifest['metadata_tar']
DATA_TAR_LOCAL = PACKAGE_LOCAL_ROOT / package_manifest['data_tar']

for path in [PROJECT_LOCAL, PACKAGE_LOCAL_ROOT, EVAL_LOCAL, SOURCE_RUN_LOCAL, COMPARISON_LOCAL, METRICS_LOCAL]:
    if path.exists():
        shutil.rmtree(path)
if FORCE_REBUILD_COMPARISON and COMPARISON_DRIVE.exists():
    shutil.rmtree(COMPARISON_DRIVE)
if FORCE_REBUILD_METRICS and METRICS_DRIVE.exists():
    shutil.rmtree(METRICS_DRIVE)

PACKAGE_LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
shutil.copytree(REPO_DRIVE, PROJECT_LOCAL)
shutil.copytree(SOURCE_RUN_DRIVE, SOURCE_RUN_LOCAL)
shutil.copy2(METADATA_TAR_DRIVE, METADATA_TAR_LOCAL)
shutil.copy2(DATA_TAR_DRIVE, DATA_TAR_LOCAL)

for command in [
    ['tar', '-xf', str(METADATA_TAR_LOCAL), '-C', '/content'],
    ['tar', '-xf', str(DATA_TAR_LOCAL), '-C', '/content'],
]:
    print('Running extract command:', ' '.join(command))
    subprocess.run(command, check=True)

assert EVAL_LOCAL.exists(), EVAL_LOCAL
print('Copied repo, source run, and extracted 32-sample eval package.')
"""
        ),
        code(
            """import json

layered_manifest_path = EVAL_LOCAL / 'metadata' / 'layered_samples.jsonl'
rows = [json.loads(line) for line in layered_manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(rows) == MAX_COMPARE_SAMPLES, len(rows)
assert sorted(set(row['split'] for row in rows)) == ['test']
assert all(row['layer_names'] == ['BG', 'HAIR', 'FACE'] for row in rows), rows[0]
assert all(row['layer_count'] == 3 for row in rows), rows[0]

for variant in MODEL_VARIANTS:
    if variant['type'] == 'base':
        continue
    checkpoint_dir = variant['run_dir'] / variant['checkpoint_name']
    assert checkpoint_dir.exists(), checkpoint_dir
    assert (checkpoint_dir / 'pytorch_lora_weights.safetensors').exists(), checkpoint_dir

print('Selected eval sample ids:', [int(row['sample_id']) for row in rows])
print('Model variants:', [variant['name'] for variant in MODEL_VARIANTS])
"""
        ),
        code(
            """import json
import shutil
import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image

assert torch.cuda.is_available(), 'This comparison requires a GPU runtime.'

comparison_summary = []
if COMPARISON_LOCAL.exists():
    shutil.rmtree(COMPARISON_LOCAL)
COMPARISON_LOCAL.mkdir(parents=True, exist_ok=True)

for model_variant in MODEL_VARIANTS:
    model_name = model_variant['name']
    checkpoint_dir = (
        model_variant.get('run_dir', SOURCE_RUN_LOCAL) / model_variant['checkpoint_name']
        if model_variant['type'] == 'lora'
        else None
    )
    source_label = checkpoint_dir if checkpoint_dir is not None else 'Qwen/Qwen-Image-Layered base model'
    print(f'Running default-call inference for {model_name} from {source_label}')
    torch.cuda.empty_cache()
    pipeline = QwenImageLayeredPipeline.from_pretrained('Qwen/Qwen-Image-Layered')
    pipeline = pipeline.to('cuda', torch.bfloat16)
    pipeline.set_progress_bar_config(disable=None)
    if checkpoint_dir is not None:
        pipeline.load_lora_weights(str(checkpoint_dir), weight_name='pytorch_lora_weights.safetensors')

    for row in rows:
        sample_id = f\"{int(row['sample_id']):05d}\"
        composite_path = EVAL_LOCAL / row['composite_path']
        input_image = Image.open(composite_path).convert('RGBA')
        requested_layers = max(1, min(int(row.get('layer_count') or 3), MAX_LAYERS))

        sample_output_dir = COMPARISON_LOCAL / model_name / sample_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        input_image.save(sample_output_dir / 'input.png')
        (sample_output_dir / 'selected_sample.json').write_text(json.dumps(row, indent=2, sort_keys=True), encoding='utf-8')
        (sample_output_dir / 'variant.json').write_text(
            json.dumps({
                'name': model_variant['name'],
                'type': model_variant['type'],
                'checkpoint_name': model_variant.get('checkpoint_name'),
            }, indent=2, sort_keys=True),
            encoding='utf-8',
        )

        call_kwargs = {
            'image': input_image,
            'generator': torch.Generator(device='cuda').manual_seed(SEED),
            'true_cfg_scale': TRUE_CFG_SCALE,
            'negative_prompt': ' ',
            'num_inference_steps': INFERENCE_STEPS,
            'num_images_per_prompt': 1,
            'layers': requested_layers,
            'resolution': RESOLUTION,
            'cfg_normalize': True,
            'use_en_prompt': True,
        }

        with torch.inference_mode():
            output = pipeline(**call_kwargs)

        predicted_layers = output.images[0]
        if len(predicted_layers) != requested_layers:
            raise RuntimeError(f'Sample {sample_id} requested {requested_layers} layers but got {len(predicted_layers)}.')

        saved_layer_paths = []
        for layer_index, image in enumerate(predicted_layers):
            output_path = sample_output_dir / f'layer_{layer_index:02d}.png'
            image.save(output_path)
            saved_layer_paths.append(str(output_path))

        comparison_summary.append({
            'model_name': model_name,
            'model_type': model_variant['type'],
            'checkpoint_name': model_variant.get('checkpoint_name'),
            'sample_id': sample_id,
            'requested_layers': requested_layers,
            'predicted_layers': len(saved_layer_paths),
            'output_dir': str(sample_output_dir),
        })
        print(f'  sample {sample_id}: requested={requested_layers} predicted={len(saved_layer_paths)}')

    del pipeline
    torch.cuda.empty_cache()

(COMPARISON_LOCAL / 'comparison_summary.json').write_text(
    json.dumps(comparison_summary, indent=2, sort_keys=True),
    encoding='utf-8',
)

RUNS_DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
if COMPARISON_DRIVE.exists():
    shutil.rmtree(COMPARISON_DRIVE)
shutil.copytree(COMPARISON_LOCAL, COMPARISON_DRIVE)

print('Default-call 32-sample comparison finished.')
print('Saved local outputs to:', COMPARISON_LOCAL)
print('Saved Drive outputs to:', COMPARISON_DRIVE)
"""
        ),
        code(
            """import shutil
import subprocess
import sys

if METRICS_LOCAL.exists():
    shutil.rmtree(METRICS_LOCAL)

command = [
    sys.executable,
    str(PROJECT_LOCAL / 'scripts' / 'evaluate_qwen_layered_outputs.py'),
    '--comparison-root',
    str(COMPARISON_LOCAL),
    '--eval-root',
    str(EVAL_LOCAL),
    '--output-root',
    str(METRICS_LOCAL),
    '--force',
]
print('Running metrics command:', ' '.join(command))
subprocess.run(command, check=True)

if METRICS_DRIVE.exists():
    shutil.rmtree(METRICS_DRIVE)
shutil.copytree(METRICS_LOCAL, METRICS_DRIVE)

print('Saved metric outputs to:', METRICS_DRIVE)
print('Summary CSV:', METRICS_DRIVE / 'metrics_summary_by_variant.csv')
"""
        ),
        code(
            """import csv
import json

summary_rows = list(csv.DictReader((METRICS_LOCAL / 'metrics_summary_by_variant.csv').open('r', encoding='utf-8')))
print(json.dumps(summary_rows, indent=2))

cols = [
    'variant_name',
    'sample_count',
    'macro_iou',
    'macro_soft_iou',
    'macro_alpha_mae',
    'empty_layer_rate',
    'full_layer_rate',
    'multi_layer_overlap_fraction',
    'bg_iou',
    'hair_iou',
    'face_iou',
]
print('| ' + ' | '.join(cols) + ' |')
print('| ' + ' | '.join(['---'] * len(cols)) + ' |')
for row in summary_rows:
    cells = []
    for col in cols:
        value = row.get(col, '')
        try:
            cells.append(f'{float(value):.4f}')
        except (TypeError, ValueError):
            cells.append(str(value))
    print('| ' + ' | '.join(cells) + ' |')
"""
        ),
        code(
            """import csv
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

FIGURE_LOCAL = METRICS_LOCAL / 'figures'
FIGURE_LOCAL.mkdir(parents=True, exist_ok=True)

def read_csv_rows(path):
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def as_float(row, key, default=np.nan):
    value = row.get(key)
    if value in (None, ''):
        return default
    try:
        return float(value)
    except ValueError:
        return default

summary_rows = read_csv_rows(METRICS_LOCAL / 'metrics_summary_by_variant.csv')
layer_rows = read_csv_rows(METRICS_LOCAL / 'metrics_per_sample_layer.csv')

variant_names = [row['variant_name'] for row in summary_rows]
display_names = [name.replace('default__', '').replace('_', ' ') for name in variant_names]
colors = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
summary_specs = [
    ('macro_iou', 'Macro IoU', 'higher is better'),
    ('macro_soft_iou', 'Soft Alpha IoU', 'higher is better'),
    ('macro_alpha_mae', 'Alpha MAE', 'lower is better'),
]
for ax, (metric_key, title, subtitle) in zip(axes, summary_specs):
    values = [as_float(row, metric_key) for row in summary_rows]
    ax.bar(display_names, values, color=colors[:len(values)])
    ax.set_title(f'{title}\\n{subtitle}', fontsize=11)
    ax.set_ylim(0, max(1.0, np.nanmax(values) * 1.15 if values else 1.0))
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(axis='y', alpha=0.25)
fig.suptitle('BG / HAIR / FACE Default-Call Evaluation Summary', fontsize=14)
fig.tight_layout()
summary_figure = FIGURE_LOCAL / 'summary_metrics.png'
fig.savefig(summary_figure, dpi=160, bbox_inches='tight')
plt.show()

layer_metric = defaultdict(lambda: defaultdict(list))
for row in layer_rows:
    layer_metric[row['variant_name']][row['layer_name']].append(as_float(row, 'iou'))

layer_names = ['BG', 'HAIR', 'FACE']
x = np.arange(len(layer_names))
width = 0.8 / max(1, len(variant_names))
fig, ax = plt.subplots(figsize=(9, 4.5))
for idx, variant_name in enumerate(variant_names):
    means = [np.nanmean(layer_metric[variant_name][layer]) for layer in layer_names]
    offset = (idx - (len(variant_names) - 1) / 2) * width
    ax.bar(x + offset, means, width=width, label=display_names[idx], color=colors[idx % len(colors)])
ax.set_title('Per-Layer IoU Against CelebAMask-Derived Targets')
ax.set_xticks(x)
ax.set_xticklabels(layer_names)
ax.set_ylim(0, 1)
ax.set_ylabel('IoU')
ax.grid(axis='y', alpha=0.25)
ax.legend()
fig.tight_layout()
layer_figure = FIGURE_LOCAL / 'per_layer_iou.png'
fig.savefig(layer_figure, dpi=160, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.5))
diagnostic_keys = ['empty_layer_rate', 'full_layer_rate', 'multi_layer_overlap_fraction']
diagnostic_labels = ['Empty layers', 'Full layers', 'Overlap']
x = np.arange(len(diagnostic_keys))
for idx, row in enumerate(summary_rows):
    values = [as_float(row, key) for key in diagnostic_keys]
    offset = (idx - (len(summary_rows) - 1) / 2) * width
    ax.bar(x + offset, values, width=width, label=display_names[idx], color=colors[idx % len(colors)])
ax.set_title('Layer Health Diagnostics')
ax.set_xticks(x)
ax.set_xticklabels(diagnostic_labels)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.25)
ax.legend()
fig.tight_layout()
diagnostic_figure = FIGURE_LOCAL / 'layer_health_diagnostics.png'
fig.savefig(diagnostic_figure, dpi=160, bbox_inches='tight')
plt.show()

drive_figure_dir = METRICS_DRIVE / 'figures'
if drive_figure_dir.exists():
    shutil.rmtree(drive_figure_dir)
shutil.copytree(FIGURE_LOCAL, drive_figure_dir)

print('Saved figures:')
for path in [summary_figure, layer_figure, diagnostic_figure]:
    print(' -', path)
print('Copied figures to:', drive_figure_dir)
"""
        ),
        code(
            """import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONTACT_LOCAL = METRICS_LOCAL / 'qualitative_contact_sheets'
CONTACT_LOCAL.mkdir(parents=True, exist_ok=True)

LAYER_NAMES = ['BG', 'HAIR', 'FACE']
MODEL_ORDER = ['base_model', 'checkpoint_50']
MODEL_LABELS = {'base_model': 'Base', 'checkpoint_50': 'LoRA ckpt-50'}

def read_rows(path):
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def metadata_path(path_str):
    return Path(str(path_str).replace('\\\\', '/').strip('/'))

def rgba_on_checker(path, size=(128, 128)):
    image = Image.open(path).convert('RGBA').resize(size, Image.Resampling.BICUBIC)
    tile = 16
    bg = Image.new('RGBA', size, (240, 240, 240, 255))
    draw = ImageDraw.Draw(bg)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=(210, 210, 210, 255))
    bg.alpha_composite(image)
    return bg.convert('RGB')

def rgb_thumb(path, size=(128, 128)):
    return Image.open(path).convert('RGB').resize(size, Image.Resampling.BICUBIC)

def add_label(draw, xy, text):
    draw.text(xy, text, fill=(20, 20, 20))

layer_rows = read_rows(METRICS_LOCAL / 'metrics_per_sample_layer.csv')
by_sample_variant = defaultdict(lambda: defaultdict(dict))
for row in layer_rows:
    by_sample_variant[row['sample_id_str']][row['variant_name']][row['layer_name']] = safe_float(row['iou'])

sample_scores = []
for sample_id, variants in by_sample_variant.items():
    if not all(name in variants for name in MODEL_ORDER):
        continue
    base_macro = np.nanmean([variants['base_model'].get(layer, np.nan) for layer in LAYER_NAMES])
    ckpt_macro = np.nanmean([variants['checkpoint_50'].get(layer, np.nan) for layer in LAYER_NAMES])
    sample_scores.append({
        'sample_id': sample_id,
        'base_macro_iou': float(base_macro),
        'checkpoint_macro_iou': float(ckpt_macro),
        'delta_macro_iou': float(ckpt_macro - base_macro),
    })

sample_scores = sorted(sample_scores, key=lambda row: row['delta_macro_iou'])
selected = []
if sample_scores:
    selected.extend(sample_scores[:2])
    selected.extend(sample_scores[-2:])
    selected.append(sample_scores[len(sample_scores) // 2])

deduped = {}
for row in selected:
    deduped[row['sample_id']] = row
selected = sorted(deduped.values(), key=lambda row: row['sample_id'])

thumb = (128, 128)
label_h = 28
pad = 12
cols = 10
rows_per_sheet = 1
sheet_w = pad + cols * (thumb[0] + pad)
sheet_h = pad + label_h + rows_per_sheet * (thumb[1] + label_h + pad)

summary = []
for score in selected:
    sample_id = score['sample_id']
    base_dir = COMPARISON_LOCAL / 'base_model' / sample_id
    ckpt_dir = COMPARISON_LOCAL / 'checkpoint_50' / sample_id
    selected_sample = json.loads((base_dir / 'selected_sample.json').read_text(encoding='utf-8'))

    columns = []
    columns.append(('Input', rgb_thumb(base_dir / 'input.png', thumb)))
    for idx, layer_name in enumerate(LAYER_NAMES):
        target_path = EVAL_LOCAL / metadata_path(selected_sample['layer_paths'][idx])
        columns.append((f'Target {layer_name}', rgba_on_checker(target_path, thumb)))
    for model_name, model_dir in [('Base', base_dir), ('LoRA', ckpt_dir)]:
        for idx, layer_name in enumerate(LAYER_NAMES):
            columns.append((f'{model_name} {layer_name}', rgba_on_checker(model_dir / f'layer_{idx:02d}.png', thumb)))

    sheet = Image.new('RGB', (sheet_w, sheet_h + 34), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title = (
        f\"sample {sample_id} | base IoU={score['base_macro_iou']:.3f} | \"
        f\"ckpt IoU={score['checkpoint_macro_iou']:.3f} | delta={score['delta_macro_iou']:+.3f}\"
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
    print(f\" - {row['sample_id']}: delta={row['delta_macro_iou']:+.4f} -> {row['contact_sheet']}\")
print('Copied contact sheets to:', drive_contact_dir)
"""
        ),
        markdown(
            """## Result

Use the printed Markdown table and `metrics_summary_by_variant.csv` for the blog/report.
The main comparison is `base_model` vs `checkpoint_50` under default-call inference.
"""
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Evaluate_BG_HAIR_FACE_5K_Default_Call_32.ipynb", cells)


def main() -> None:
    write_cpu_eval32_package_notebook()
    write_gpu_eval32_notebook()


if __name__ == "__main__":
    main()
