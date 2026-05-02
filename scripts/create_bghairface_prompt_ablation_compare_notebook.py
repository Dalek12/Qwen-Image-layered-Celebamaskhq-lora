"""Create an inference-only prompt ablation notebook for BG/HAIR/FACE checkpoint-50."""

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


def main() -> None:
    cells = [
        markdown(
            """# CelebAMask-HQ GPU BG / HAIR / FACE Prompt Ablation

Use this notebook with a GPU runtime. It does not train. It compares the base model
and the generic-prompt `checkpoint-50` across several inference prompts on the same
held-out BG/HAIR/FACE eval package.
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
PACKAGE_LOCAL_ROOT = Path('/content/test_eval_package_prompt_ablation')

EVAL_OUTPUT_NAME = 'processed_celebmaskhq_bghairface_5k_test_eval_package'
EVAL_OUTPUT_DRIVE = PROCESSED_DRIVE_ROOT / EVAL_OUTPUT_NAME
PACKAGE_MANIFEST_DRIVE = EVAL_OUTPUT_DRIVE / 'package_manifest.json'

SOURCE_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'
SOURCE_RUN_DRIVE = RUNS_DRIVE_ROOT / SOURCE_RUN_NAME
SOURCE_RUN_LOCAL = Path(f'/content/{SOURCE_RUN_NAME}')
SOURCE_CHECKPOINT_NAME = 'checkpoint-50'

COMPARISON_RUN_NAME = 'qwen_layered_lora_bghairface_5k_prompt_ablation_compare'
COMPARISON_DRIVE = RUNS_DRIVE_ROOT / COMPARISON_RUN_NAME
COMPARISON_LOCAL = Path(f'/content/{COMPARISON_RUN_NAME}')

PROMPT_VARIANTS = [
    {
        'name': 'default_call',
        'prompt': None,
        'use_en_prompt': True,
        'description': 'Do not pass prompt; let the pipeline/default behavior apply.',
    },
    {
        'name': 'generic_decompose',
        'prompt': 'decompose this portrait into editable portrait layers',
        'use_en_prompt': True,
        'description': 'Original generic training prompt.',
    },
    {
        'name': 'portrait_person_face_hair',
        'prompt': 'a close-up portrait photo of a person with visible face and hair',
        'use_en_prompt': True,
        'description': 'Content description without layer-control language.',
    },
    {
        'name': 'celebrity_face_hair_background',
        'prompt': 'a celebrity portrait photo with face, hair, and background',
        'use_en_prompt': True,
        'description': 'Slightly more dataset-specific content description.',
    },
]

MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'checkpoint_50', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': SOURCE_CHECKPOINT_NAME},
]

MAX_COMPARE_SAMPLES = 3
FOCUS_SAMPLE_IDS = {2839}
MIXED_PRECISION = 'bf16'
RESOLUTION = 640
MAX_LAYERS = 3
INFERENCE_STEPS = 50
TRUE_CFG_SCALE = 4.0
SEED = 1337
FORCE_REBUILD_COMPARISON = True

print('Repo on Drive:', REPO_DRIVE)
print('Eval package:', EVAL_OUTPUT_DRIVE)
print('Source run:', SOURCE_RUN_DRIVE)
print('Comparison output:', COMPARISON_DRIVE)
print('Prompt variants:', [variant['name'] for variant in PROMPT_VARIANTS])
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

PACKAGE_NAME = package_manifest['package_name']
METADATA_TAR_DRIVE = EVAL_OUTPUT_DRIVE / package_manifest['metadata_tar']
DATA_TAR_DRIVE = EVAL_OUTPUT_DRIVE / package_manifest['data_tar']
EVAL_LOCAL = Path('/content') / PACKAGE_NAME
METADATA_TAR_LOCAL = PACKAGE_LOCAL_ROOT / package_manifest['metadata_tar']
DATA_TAR_LOCAL = PACKAGE_LOCAL_ROOT / package_manifest['data_tar']

for path in [PROJECT_LOCAL, PACKAGE_LOCAL_ROOT, EVAL_LOCAL, SOURCE_RUN_LOCAL, COMPARISON_LOCAL]:
    if path.exists():
        shutil.rmtree(path)
if FORCE_REBUILD_COMPARISON and COMPARISON_DRIVE.exists():
    shutil.rmtree(COMPARISON_DRIVE)

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
print('Copied repo, source run, and extracted eval package.')
"""
        ),
        code(
            """import json

layered_manifest_path = EVAL_LOCAL / 'metadata' / 'layered_samples.jsonl'
all_rows = [json.loads(line) for line in layered_manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert sorted(set(row['split'] for row in all_rows)) == ['test']
assert all(row['layer_names'] == ['BG', 'HAIR', 'FACE'] for row in all_rows), all_rows[0]

focused_rows = [row for row in all_rows if int(row['sample_id']) in FOCUS_SAMPLE_IDS]
remaining_rows = [row for row in all_rows if int(row['sample_id']) not in FOCUS_SAMPLE_IDS]
if len(focused_rows) >= MAX_COMPARE_SAMPLES:
    rows = focused_rows[:MAX_COMPARE_SAMPLES]
else:
    extra_needed = MAX_COMPARE_SAMPLES - len(focused_rows)
    if len(remaining_rows) <= extra_needed:
        extras = remaining_rows
    elif extra_needed <= 0:
        extras = []
    elif extra_needed == 1:
        extras = [remaining_rows[len(remaining_rows) // 2]]
    else:
        last_index = len(remaining_rows) - 1
        extras = [remaining_rows[round(i * last_index / (extra_needed - 1))] for i in range(extra_needed)]
    rows = focused_rows + extras

for variant in MODEL_VARIANTS:
    if variant['type'] == 'base':
        continue
    checkpoint_dir = variant['run_dir'] / variant['checkpoint_name']
    assert checkpoint_dir.exists(), checkpoint_dir
    assert (checkpoint_dir / 'pytorch_lora_weights.safetensors').exists(), checkpoint_dir

print('Selected compare sample ids:', [int(row['sample_id']) for row in rows])
print('Model variants:', [variant['name'] for variant in MODEL_VARIANTS])
"""
        ),
        code(
            """import json
import shutil
import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image

assert torch.cuda.is_available(), 'This prompt ablation requires a GPU runtime.'

comparison_summary = []
if COMPARISON_LOCAL.exists():
    shutil.rmtree(COMPARISON_LOCAL)
COMPARISON_LOCAL.mkdir(parents=True, exist_ok=True)

for prompt_variant in PROMPT_VARIANTS:
    prompt_name = prompt_variant['name']
    prompt = prompt_variant['prompt']
    for model_variant in MODEL_VARIANTS:
        model_name = model_variant['name']
        variant_name = f'{prompt_name}__{model_name}'
        checkpoint_dir = (
            model_variant.get('run_dir', SOURCE_RUN_LOCAL) / model_variant['checkpoint_name']
            if model_variant['type'] == 'lora'
            else None
        )
        source_label = checkpoint_dir if checkpoint_dir is not None else 'Qwen/Qwen-Image-Layered base model'
        print(f'Running {variant_name} from {source_label}')
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

            sample_output_dir = COMPARISON_LOCAL / prompt_name / model_name / sample_id
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            input_image.save(sample_output_dir / 'input.png')
            (sample_output_dir / 'selected_sample.json').write_text(json.dumps(row, indent=2, sort_keys=True), encoding='utf-8')
            (sample_output_dir / 'variant.json').write_text(
                json.dumps({
                    'prompt_variant': prompt_variant,
                    'model_variant': {
                        'name': model_variant['name'],
                        'type': model_variant['type'],
                        'checkpoint_name': model_variant.get('checkpoint_name'),
                    },
                    'variant_name': variant_name,
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
                'use_en_prompt': prompt_variant['use_en_prompt'],
            }
            if prompt is not None:
                call_kwargs['prompt'] = prompt

            with torch.inference_mode():
                output = pipeline(**call_kwargs)

            predicted_layers = output.images[0]
            saved_layer_paths = []
            for layer_index, image in enumerate(predicted_layers):
                output_path = sample_output_dir / f'layer_{layer_index:02d}.png'
                image.save(output_path)
                saved_layer_paths.append(str(output_path))

            comparison_summary.append({
                'prompt_name': prompt_name,
                'prompt': prompt,
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

print('Prompt ablation finished.')
print('Saved local outputs to:', COMPARISON_LOCAL)
print('Saved Drive outputs to:', COMPARISON_DRIVE)
"""
        ),
        code(
            """for row in comparison_summary:
    print(json.dumps(row, indent=2, sort_keys=True))
"""
        ),
        markdown(
            """## Result

Review the saved comparison directories on Drive to choose the best inference prompt
before deciding whether another training run is worthwhile.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Prompt_Ablation.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
