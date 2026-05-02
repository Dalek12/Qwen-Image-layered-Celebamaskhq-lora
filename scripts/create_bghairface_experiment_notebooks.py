"""Create Colab notebooks for the BG/HAIR/FACE 5k experiment."""

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


def write_cpu_preprocess_package_notebook() -> None:
    cells = [
        markdown(
            """# CelebAMask-HQ CPU BG / HAIR / FACE 5k Preprocess And Package

Use this notebook with a CPU runtime. It builds the 5k `BG / HAIR / FACE`
processed dataset, verifies the metadata, and creates both the train/val shard
package and the held-out test eval package on Drive.
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
RAW_DRIVE = DRIVE_ROOT / 'raw'
PROCESSED_DRIVE_ROOT = DRIVE_ROOT / 'processed'
PROJECT_LOCAL = Path('/content/project')

PROCESSED_NAME = 'processed_celebmaskhq_bghairface_5k'
TRAINVAL_PACKAGE_NAME = 'processed_celebmaskhq_bghairface_5k_trainval'
TRAINVAL_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_trainval_shards'
EVAL_PACKAGE_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval'
EVAL_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval_package'

PROCESSED_DRIVE = PROCESSED_DRIVE_ROOT / PROCESSED_NAME
TRAINVAL_OUTPUT_DRIVE = PROCESSED_DRIVE_ROOT / TRAINVAL_OUTPUT_NAME
EVAL_OUTPUT_DRIVE = PROCESSED_DRIVE_ROOT / EVAL_OUTPUT_NAME

LAYERED_SCHEME = 'bg_hair_face'
SAMPLE_LIMIT = 5000
PREVIEW_COUNT = 8
MAX_LAYERS = 3
SHARD_SIZE = 2000
MAX_EVAL_SAMPLES = 8
PROGRESS_EVERY = 100

FORCE_REBUILD_PROCESSED = False
RESUME_EXISTING_PROCESSED = True
FORCE_REBUILD_TRAINVAL_PACKAGE = False
RESUME_EXISTING_TRAINVAL_PACKAGE = True
FORCE_REBUILD_EVAL_PACKAGE = False

print('Repo on Drive:', REPO_DRIVE)
print('Raw data on Drive:', RAW_DRIVE)
print('Processed output:', PROCESSED_DRIVE)
print('Train/val package:', TRAINVAL_OUTPUT_DRIVE)
print('Test eval package:', EVAL_OUTPUT_DRIVE)
"""
        ),
        code("!pip install -U pillow numpy torch matplotlib\n"),
        code(
            """import shutil

assert REPO_DRIVE.exists(), f'Missing repo folder on Drive: {REPO_DRIVE}'
assert RAW_DRIVE.exists(), f'Missing raw folder on Drive: {RAW_DRIVE}'
assert (RAW_DRIVE / 'CelebA-HQ-img').exists(), 'Missing raw/CelebA-HQ-img on Drive'
assert (RAW_DRIVE / 'CelebAMask-HQ-mask-anno').exists(), 'Missing raw/CelebAMask-HQ-mask-anno on Drive'

if PROJECT_LOCAL.exists():
    shutil.rmtree(PROJECT_LOCAL)
shutil.copytree(REPO_DRIVE, PROJECT_LOCAL)
PROCESSED_DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print('Copied repo to', PROJECT_LOCAL)
"""
        ),
        code(
            """import subprocess

def run_checked(command, cwd, capture_output=True):
    printable = ' '.join(str(part) for part in command)
    print('Running command:', printable)
    if capture_output:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
        print('RETURN CODE:', result.returncode)
        print('--- STDOUT ---')
        print(result.stdout)
        print('--- STDERR ---')
        print(result.stderr)
    else:
        result = subprocess.run(command, cwd=cwd, text=True)
        print('RETURN CODE:', result.returncode)
    if result.returncode != 0:
        raise RuntimeError('Command failed; see stdout/stderr above.')
    return result

pipeline_text = (PROJECT_LOCAL / 'preprocess_celebmaskhq' / 'pipeline.py').read_text(encoding='utf-8')
freshness_markers = ['LAYERED_SCHEMES', 'bg_hair_face', 'processed_splits.json', 'source_discovery_cache.json', 'layer_source_slots']
missing_markers = [marker for marker in freshness_markers if marker not in pipeline_text]
assert not missing_markers, f'Copied repo is stale; missing markers: {missing_markers}'
print('Pipeline freshness check passed.')
"""
        ),
        code(
            """import json
import shutil
import sys

if FORCE_REBUILD_PROCESSED and PROCESSED_DRIVE.exists():
    shutil.rmtree(PROCESSED_DRIVE)

processed_metadata_path = PROCESSED_DRIVE / 'metadata' / 'layered_samples.jsonl'

def processed_metadata_matches_target():
    if not processed_metadata_path.exists():
        return False
    try:
        mapping = json.loads((PROCESSED_DRIVE / 'metadata' / 'mapping.json').read_text(encoding='utf-8'))
        stats = json.loads((PROCESSED_DRIVE / 'metadata' / 'stats.json').read_text(encoding='utf-8'))
        rows = [json.loads(line) for line in processed_metadata_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception as exc:
        print('Existing metadata could not be read, will rebuild/resume:', repr(exc))
        return False
    mismatched_rows = [
        row for row in rows
        if row.get('layer_names') != ['BG', 'HAIR', 'FACE'] or int(row.get('layer_count', -1)) != 3
    ]
    if mismatched_rows:
        print('Existing metadata has rows outside the fixed BG/HAIR/FACE 3-layer contract.')
        print('First mismatched row:', json.dumps(mismatched_rows[0], indent=2, sort_keys=True)[:2000])
        return False
    return (
        mapping.get('layered_export', {}).get('scheme') == LAYERED_SCHEME
        and mapping.get('layered_export', {}).get('target_layer_names') == ['BG', 'HAIR', 'FACE']
        and int(stats.get('processed_sample_count', -1)) == SAMPLE_LIMIT
        and len(rows) == SAMPLE_LIMIT
    )

if FORCE_REBUILD_PROCESSED or not processed_metadata_matches_target():
    command = [
        sys.executable, '-u', 'scripts/build_processed_celebmaskhq.py',
        '--dataset-root', str(RAW_DRIVE),
        '--output-root', str(PROCESSED_DRIVE),
        '--layered-scheme', LAYERED_SCHEME,
        '--limit', str(SAMPLE_LIMIT),
        '--preview-count', str(PREVIEW_COUNT),
        '--progress-every', str(PROGRESS_EVERY),
    ]
    if RESUME_EXISTING_PROCESSED:
        command.append('--resume-existing')
    run_checked(command, PROJECT_LOCAL, capture_output=False)
else:
    print('Processed BG/HAIR/FACE 5k dataset already exists and matches the fixed 3-layer contract; skipping rebuild.')

assert processed_metadata_path.exists(), f'Missing layered metadata: {processed_metadata_path}'
print('Processed dataset ready at', PROCESSED_DRIVE)
"""
        ),
        code(
            """import json

mapping = json.loads((PROCESSED_DRIVE / 'metadata' / 'mapping.json').read_text(encoding='utf-8'))
stats = json.loads((PROCESSED_DRIVE / 'metadata' / 'stats.json').read_text(encoding='utf-8'))
rows = [json.loads(line) for line in (PROCESSED_DRIVE / 'metadata' / 'layered_samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

assert mapping['layered_export']['scheme'] == LAYERED_SCHEME
assert mapping['layered_export']['target_layer_names'] == ['BG', 'HAIR', 'FACE']
assert stats['processed_sample_count'] == SAMPLE_LIMIT, stats['processed_sample_count']
mismatched_rows = [
    row for row in rows
    if row.get('layer_names') != ['BG', 'HAIR', 'FACE'] or int(row.get('layer_count', -1)) != 3
]
assert not mismatched_rows, mismatched_rows[0]

print('Metadata verification passed.')
print('Processed split counts:', stats['split_counts_processed_subset'])
print('Layer count histogram:', stats['layered_layer_count_histogram'])
"""
        ),
        code(
            """import sys

run_checked([
    sys.executable,
    'scripts/inspect_generic_layered_loader.py',
    '--processed-root', str(PROCESSED_DRIVE),
    '--split', 'train',
    '--batch-size', '1',
    '--num-batches', '1',
    '--max-layers', str(MAX_LAYERS),
    '--drop-warning-samples',
], PROJECT_LOCAL)
"""
        ),
        code(
            """import shutil
import sys

if FORCE_REBUILD_TRAINVAL_PACKAGE and TRAINVAL_OUTPUT_DRIVE.exists():
    shutil.rmtree(TRAINVAL_OUTPUT_DRIVE)

trainval_manifest_path = TRAINVAL_OUTPUT_DRIVE / 'package_manifest.json'
if not trainval_manifest_path.exists():
    command = [
        sys.executable, '-u', 'scripts/package_qwen_layered_trainval_shards.py',
        '--processed-root', str(PROCESSED_DRIVE),
        '--output-root', str(TRAINVAL_OUTPUT_DRIVE),
        '--package-name', TRAINVAL_PACKAGE_NAME,
        '--splits', 'train', 'val',
        '--shard-size', str(SHARD_SIZE),
        '--progress-every', str(PROGRESS_EVERY),
    ]
    if FORCE_REBUILD_TRAINVAL_PACKAGE:
        command.append('--force')
    elif RESUME_EXISTING_TRAINVAL_PACKAGE:
        command.append('--resume-existing')
    run_checked(command, PROJECT_LOCAL, capture_output=False)
else:
    print('Train/val shard package already exists; skipping rebuild.')

assert trainval_manifest_path.exists(), f'Missing train/val package manifest: {trainval_manifest_path}'
print('Train/val shard package ready at', TRAINVAL_OUTPUT_DRIVE)
"""
        ),
        code(
            """import json

trainval_manifest = json.loads((TRAINVAL_OUTPUT_DRIVE / 'package_manifest.json').read_text(encoding='utf-8'))
assert trainval_manifest['included_splits'] == ['train', 'val']
assert trainval_manifest['test_split_packaged'] == 0
assert trainval_manifest['data_shards'], 'No train/val data shards were produced.'
print('Train/val package verification passed.')
print('Split counts:', trainval_manifest['split_counts'])
print('Sample count:', trainval_manifest['sample_count'])
print('Data shard count:', len(trainval_manifest['data_shards']))
"""
        ),
        code(
            """import shutil
import sys

if FORCE_REBUILD_EVAL_PACKAGE and EVAL_OUTPUT_DRIVE.exists():
    shutil.rmtree(EVAL_OUTPUT_DRIVE)

eval_manifest_path = EVAL_OUTPUT_DRIVE / 'package_manifest.json'
if not eval_manifest_path.exists():
    command = [
        sys.executable, '-u', 'scripts/package_qwen_layered_eval_subset.py',
        '--processed-root', str(PROCESSED_DRIVE),
        '--output-root', str(EVAL_OUTPUT_DRIVE),
        '--package-name', EVAL_PACKAGE_NAME,
        '--split', 'test',
        '--max-samples', str(MAX_EVAL_SAMPLES),
        '--sample-strategy', 'spaced',
        '--progress-every', '1',
    ]
    if FORCE_REBUILD_EVAL_PACKAGE:
        command.append('--force')
    run_checked(command, PROJECT_LOCAL, capture_output=False)
else:
    print('Test eval package already exists; skipping rebuild.')

assert eval_manifest_path.exists(), f'Missing test eval package manifest: {eval_manifest_path}'
print('Test eval package ready at', EVAL_OUTPUT_DRIVE)
"""
        ),
        code(
            """import json

eval_manifest = json.loads((EVAL_OUTPUT_DRIVE / 'package_manifest.json').read_text(encoding='utf-8'))
assert eval_manifest['target_split'] == 'test'
assert eval_manifest['sample_count'] <= MAX_EVAL_SAMPLES
print('Test eval package verification passed.')
print('Selected test sample ids:', eval_manifest['selected_sample_ids'])
"""
        ),
        markdown(
            """## Result

The 5k `BG / HAIR / FACE` processed dataset and both Colab packages are ready on Drive:

- `processed/processed_celebmaskhq_bghairface_5k`
- `processed/processed_celebmaskhq_bghairface_5k_trainval_shards`
- `processed/processed_celebmaskhq_bghairface_5k_test_eval_package`
"""
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "CelebAMaskHQ_CPU_BG_HAIR_FACE_5K_Preprocess_And_Package.ipynb", cells)


def write_variant_notebooks() -> None:
    train_nb = json.loads((NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_TrainVal_Shards_Micro_Ablation.ipynb").read_text(encoding="utf-8"))
    train_nb["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Training BG / HAIR / FACE 5k Micro Ablation

Use this notebook with a GPU runtime after the BG/HAIR/FACE CPU notebook creates the train/val shard package.
It keeps the alpha-weighted micro-ablation recipe and changes only the target representation.
"""
    )["source"]
    for cell in train_nb["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace("SHARD_OUTPUT_NAME = 'processed_celebmaskhq_trainval_shards'", "SHARD_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_trainval_shards'")
        text = text.replace("RUN_NAME = 'qwen_layered_lora_trainval_micro_ablation_alpha_weighted'", "RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'")
        text = text.replace("MAX_LAYERS = 8", "MAX_LAYERS = 3")
        text = text.replace("qwen_layered_lora_trainval_micro_ablation_alpha_weighted", "qwen_layered_lora_bghairface_5k_micro_ablation")
        cell["source"] = text.splitlines(keepends=True)
    check = "assert all(row['split'] != 'test' for row in rows)\n"
    add = "assert all(row['layer_names'] == ['BG', 'HAIR', 'FACE'] for row in rows), rows[0]\nassert all(row['layer_count'] == 3 for row in rows), rows[0]\n"
    text = "".join(train_nb["cells"][7]["source"])
    if add not in text:
        train_nb["cells"][7]["source"] = text.replace(check, check + add).splitlines(keepends=True)
    train_nb["cells"][-1]["source"] = markdown(
        """## Result

This run trains the 5k `BG / HAIR / FACE` target scheme with LR `1e-5`, rank `8`,
LoRA alpha `8`, 100 steps, and checkpoints every 25 steps.
"""
    )["source"]
    (NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Micro_Ablation.ipynb").write_text(json.dumps(train_nb, indent=1), encoding="utf-8")

    compare_nb = json.loads((NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_Micro_Ablation_Alpha_Weighted_Checkpoints.ipynb").read_text(encoding="utf-8"))
    compare_nb["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Compare BG / HAIR / FACE 5k Micro-Ablation Checkpoints

Use this notebook with a GPU runtime after the BG/HAIR/FACE 5k micro-ablation run completes.
It compares checkpoints 25/50/75 on the held-out BG/HAIR/FACE test subset.
"""
    )["source"]
    for cell in compare_nb["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace("EVAL_OUTPUT_NAME = 'processed_celebmaskhq_test_eval_package'", "EVAL_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval_package'")
        text = text.replace("MICRO_RUN_NAME = 'qwen_layered_lora_trainval_micro_ablation_alpha_weighted'", "MICRO_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'")
        text = text.replace("COMPARISON_RUN_NAME = 'qwen_layered_lora_micro_ablation_alpha_weighted_compare'", "COMPARISON_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation_compare'")
        text = text.replace("MAX_LAYERS = 8", "MAX_LAYERS = 3")
        cell["source"] = text.splitlines(keepends=True)
    cell2_text = "".join(compare_nb["cells"][2]["source"])
    base_variant = (
        "MODEL_VARIANTS = [\n"
        "    {\n"
        "        'name': 'base_model',\n"
        "        'type': 'base',\n"
        "    },\n"
    )
    cell2_text = cell2_text.replace("MODEL_VARIANTS = [\n", base_variant, 1)
    compare_nb["cells"][2]["source"] = cell2_text.splitlines(keepends=True)
    check = "assert row_splits == ['test'], row_splits\n"
    add = "assert all(row['layer_names'] == ['BG', 'HAIR', 'FACE'] for row in all_rows), all_rows[0]\nassert all(row['layer_count'] == 3 for row in all_rows), all_rows[0]\n"
    text = "".join(compare_nb["cells"][6]["source"])
    if add not in text:
        text = text.replace(check, check + add)
    old_checkpoints = (
        "for variant in MODEL_VARIANTS:\n"
        "    checkpoint_dir = variant['run_dir'] / variant['checkpoint_name']\n"
        "    assert checkpoint_dir.exists(), f'Missing checkpoint directory: {checkpoint_dir}'\n"
        "    assert (checkpoint_dir / 'pytorch_lora_weights.safetensors').exists(), checkpoint_dir\n"
    )
    new_checkpoints = (
        "for variant in MODEL_VARIANTS:\n"
        "    if variant['type'] == 'base':\n"
        "        continue\n"
        "    checkpoint_dir = variant['run_dir'] / variant['checkpoint_name']\n"
        "    assert checkpoint_dir.exists(), f'Missing checkpoint directory: {checkpoint_dir}'\n"
        "    assert (checkpoint_dir / 'pytorch_lora_weights.safetensors').exists(), checkpoint_dir\n"
    )
    text = text.replace(old_checkpoints, new_checkpoints)
    compare_nb["cells"][6]["source"] = text.splitlines(keepends=True)
    text = "".join(compare_nb["cells"][7]["source"])
    old_setup = (
        "for variant in MODEL_VARIANTS:\n"
        "    variant_name = variant['name']\n"
        "    checkpoint_dir = variant['run_dir'] / variant['checkpoint_name']\n"
        "    print(f\"Running inference for {variant_name} from {checkpoint_dir}\")\n"
        "    torch.cuda.empty_cache()\n"
        "    pipeline = QwenImageLayeredPipeline.from_pretrained('Qwen/Qwen-Image-Layered')\n"
        "    pipeline = pipeline.to('cuda', torch.bfloat16)\n"
        "    pipeline.set_progress_bar_config(disable=None)\n"
        "    pipeline.load_lora_weights(str(checkpoint_dir), weight_name='pytorch_lora_weights.safetensors')\n"
    )
    new_setup = (
        "for variant in MODEL_VARIANTS:\n"
        "    variant_name = variant['name']\n"
        "    checkpoint_dir = variant.get('run_dir', MICRO_RUN_LOCAL) / variant['checkpoint_name'] if variant['type'] == 'lora' else None\n"
        "    source_label = checkpoint_dir if checkpoint_dir is not None else 'Qwen/Qwen-Image-Layered base model'\n"
        "    print(f\"Running inference for {variant_name} from {source_label}\")\n"
        "    torch.cuda.empty_cache()\n"
        "    pipeline = QwenImageLayeredPipeline.from_pretrained('Qwen/Qwen-Image-Layered')\n"
        "    pipeline = pipeline.to('cuda', torch.bfloat16)\n"
        "    pipeline.set_progress_bar_config(disable=None)\n"
        "    if checkpoint_dir is not None:\n"
        "        pipeline.load_lora_weights(str(checkpoint_dir), weight_name='pytorch_lora_weights.safetensors')\n"
    )
    text = text.replace(old_setup, new_setup)
    text = text.replace("'checkpoint_name': variant['checkpoint_name'],", "'checkpoint_name': variant.get('checkpoint_name'),")
    compare_nb["cells"][7]["source"] = text.splitlines(keepends=True)
    compare_nb["cells"][-1]["source"] = markdown(
        """## Result

Review the saved comparison directories on Drive to judge whether the broader
`BG / HAIR / FACE` target scheme improves over the base model and prior 5-layer LoRA runs.
"""
    )["source"]
    (NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Micro_Ablation_Checkpoints.ipynb").write_text(json.dumps(compare_nb, indent=1), encoding="utf-8")


def main() -> None:
    write_cpu_preprocess_package_notebook()
    write_variant_notebooks()
    print("Wrote BG/HAIR/FACE 5k experiment notebooks.")


if __name__ == "__main__":
    main()
