"""Create low-LR continuation notebooks for the BG/HAIR/FACE 5k experiment."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

SOURCE_RUN_NAME = "qwen_layered_lora_bghairface_5k_micro_ablation"
SOURCE_CHECKPOINT_NAME = "checkpoint-50"
RUN_NAME = "qwen_layered_lora_bghairface_5k_ckpt50_low_lr_decay"
COMPARE_NAME = "qwen_layered_lora_bghairface_5k_ckpt50_low_lr_decay_compare"
PROMPT = "decompose this portrait into editable portrait layers"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def write_training_notebook() -> None:
    source_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Micro_Ablation.ipynb"
    target_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Checkpoint50_Low_LR_Decay.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    notebook["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Training BG / HAIR / FACE 5k Checkpoint-50 Low-LR Decay

Use this notebook with a GPU runtime after the generic-prompt BG/HAIR/FACE run exists on Drive.
It resumes LoRA weights from `checkpoint-50`, resets optimizer/scheduler state, and continues with a lower decaying LR.
"""
    )["source"]

    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace(
            "RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'",
            f"RUN_NAME = '{RUN_NAME}'",
        )
        text = text.replace("FULL_MAX_STEPS = 100", "FULL_MAX_STEPS = 75")
        text = text.replace("CHECKPOINTING_STEPS = 25", "CHECKPOINTING_STEPS = 5")
        text = text.replace("VALIDATION_STEPS = 25", "VALIDATION_STEPS = 5")
        text = text.replace("SAVE_TOTAL_LIMIT = 6", "SAVE_TOTAL_LIMIT = 8")
        text = text.replace("LEARNING_RATE = 1e-5", "LEARNING_RATE = 3e-6")
        text = text.replace(
            "print('Micro ablation defaults: lr=1e-5, rank=8, alpha=8, grad_accum=4, max_steps=100, checkpoint_every=25')",
            "print('Continuation defaults: resume checkpoint-50, lr=3e-6, linear decay, max_steps=75, checkpoint_every=5')",
        )
        text = text.replace(
            "print('This alpha-weighted run uses a fresh RUN_NAME so prior micro-ablation checkpoints are not reused.')",
            "print('This continuation resets optimizer/scheduler state but loads LoRA weights from the generic checkpoint-50.')",
        )
        cell["source"] = text.splitlines(keepends=True)

    config_text = "".join(notebook["cells"][2]["source"])
    if "SOURCE_RUN_NAME =" not in config_text:
        config_text = config_text.replace(
            f"RUN_LOCAL = Path(f'/content/{{RUN_NAME}}')\n",
            (
                f"RUN_LOCAL = Path(f'/content/{{RUN_NAME}}')\n"
                f"SOURCE_RUN_NAME = '{SOURCE_RUN_NAME}'\n"
                f"SOURCE_CHECKPOINT_NAME = '{SOURCE_CHECKPOINT_NAME}'\n"
                "SOURCE_RUN_DRIVE = RUNS_DRIVE_ROOT / SOURCE_RUN_NAME\n"
                "SOURCE_RUN_LOCAL = Path(f'/content/{SOURCE_RUN_NAME}')\n"
            ),
        )
    if "PROMPT =" not in config_text:
        config_text = config_text.replace("MIXED_PRECISION = 'bf16'\n", f"MIXED_PRECISION = 'bf16'\nPROMPT = {PROMPT!r}\n")
    if "LR_SCHEDULER =" not in config_text:
        config_text = config_text.replace(
            "PROMPT = 'decompose this portrait into editable portrait layers'\n",
            (
                "PROMPT = 'decompose this portrait into editable portrait layers'\n"
                "LR_SCHEDULER = 'linear'\n"
                "LR_WARMUP_STEPS = 0\n"
                "LR_SCHEDULER_STEPS = 25\n"
            ),
        )
    if "print('Source run on Drive:', SOURCE_RUN_DRIVE)" not in config_text:
        config_text = config_text.replace(
            "print('Package manifest on Drive:', PACKAGE_MANIFEST_DRIVE)\n",
            "print('Package manifest on Drive:', PACKAGE_MANIFEST_DRIVE)\nprint('Source run on Drive:', SOURCE_RUN_DRIVE)\nprint('Source checkpoint:', SOURCE_CHECKPOINT_NAME)\nprint('Prompt:', PROMPT)\n",
        )
    notebook["cells"][2]["source"] = config_text.splitlines(keepends=True)

    setup_text = "".join(notebook["cells"][6]["source"])
    if "assert SOURCE_RUN_DRIVE.exists()" not in setup_text:
        setup_text = setup_text.replace(
            "assert PACKAGE_MANIFEST_DRIVE.exists(), f'Missing package manifest on Drive: {PACKAGE_MANIFEST_DRIVE}'\n",
            "assert PACKAGE_MANIFEST_DRIVE.exists(), f'Missing package manifest on Drive: {PACKAGE_MANIFEST_DRIVE}'\nassert SOURCE_RUN_DRIVE.exists(), f'Missing source run on Drive: {SOURCE_RUN_DRIVE}'\nassert (SOURCE_RUN_DRIVE / SOURCE_CHECKPOINT_NAME).exists(), f'Missing source checkpoint: {SOURCE_RUN_DRIVE / SOURCE_CHECKPOINT_NAME}'\n",
        )
    if "if SOURCE_RUN_LOCAL.exists():" not in setup_text:
        setup_text = setup_text.replace(
            "if RUN_LOCAL.exists():\n    shutil.rmtree(RUN_LOCAL)\n",
            "if RUN_LOCAL.exists():\n    shutil.rmtree(RUN_LOCAL)\nif SOURCE_RUN_LOCAL.exists():\n    shutil.rmtree(SOURCE_RUN_LOCAL)\n",
        )
    if "shutil.copytree(SOURCE_RUN_DRIVE, SOURCE_RUN_LOCAL)" not in setup_text:
        setup_text = setup_text.replace(
            "shutil.copytree(REPO_DRIVE, PROJECT_LOCAL)\n",
            "shutil.copytree(REPO_DRIVE, PROJECT_LOCAL)\nshutil.copytree(SOURCE_RUN_DRIVE, SOURCE_RUN_LOCAL)\n",
        )
    old_resume = (
        "RUNS_DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n"
        "RESUME_FROM_CHECKPOINT = None\n"
        "RESUME_CHECKPOINT_NAMES = []\n"
        "if RUN_DRIVE.exists() and RESUME_IF_AVAILABLE and not FORCE_RESTART_FULL_RUN:\n"
        "    shutil.copytree(RUN_DRIVE, RUN_LOCAL)\n"
        "    RESUME_CHECKPOINT_NAMES = sorted(\n"
        "        [path.name for path in RUN_LOCAL.iterdir() if path.is_dir() and path.name.startswith('checkpoint-')],\n"
        "        key=lambda name: int(name.split('-', 1)[1]),\n"
        "    )\n"
        "    if RESUME_CHECKPOINT_NAMES:\n"
        "        RESUME_FROM_CHECKPOINT = 'latest'\n"
    )
    new_resume = (
        "RUNS_DRIVE_ROOT.mkdir(parents=True, exist_ok=True)\n"
        "RESUME_FROM_CHECKPOINT = str(SOURCE_RUN_LOCAL / SOURCE_CHECKPOINT_NAME)\n"
        "RESUME_CHECKPOINT_NAMES = []\n"
        "if RUN_DRIVE.exists() and RESUME_IF_AVAILABLE and not FORCE_RESTART_FULL_RUN:\n"
        "    shutil.copytree(RUN_DRIVE, RUN_LOCAL)\n"
        "    RESUME_CHECKPOINT_NAMES = sorted(\n"
        "        [path.name for path in RUN_LOCAL.iterdir() if path.is_dir() and path.name.startswith('checkpoint-')],\n"
        "        key=lambda name: int(name.split('-', 1)[1]),\n"
        "    )\n"
        "    if RESUME_CHECKPOINT_NAMES:\n"
        "        RESUME_FROM_CHECKPOINT = 'latest'\n"
        "RESET_OPTIMIZER_ON_RESUME = RESUME_FROM_CHECKPOINT != 'latest'\n"
    )
    setup_text = setup_text.replace(old_resume, new_resume)
    notebook["cells"][6]["source"] = setup_text.splitlines(keepends=True)

    command_text = "".join(notebook["cells"][10]["source"])
    if "'--prompt', PROMPT," not in command_text:
        command_text = command_text.replace(
            "    '--processed-root', str(PROCESSED_LOCAL),\n",
            "    '--processed-root', str(PROCESSED_LOCAL),\n    '--prompt', PROMPT,\n",
        )
    if "'--lr-scheduler', LR_SCHEDULER," not in command_text:
        command_text = command_text.replace(
            "    '--learning-rate', str(LEARNING_RATE),\n",
            (
                "    '--learning-rate', str(LEARNING_RATE),\n"
                "    '--lr-scheduler', LR_SCHEDULER,\n"
                "    '--lr-warmup-steps', str(LR_WARMUP_STEPS),\n"
                "    '--lr-scheduler-steps', str(LR_SCHEDULER_STEPS),\n"
            ),
        )
    if "if RESET_OPTIMIZER_ON_RESUME:" not in command_text:
        command_text = command_text.replace(
            "if ALLOW_TF32:\n",
            "if RESET_OPTIMIZER_ON_RESUME:\n    train_command.append('--reset-optimizer-on-resume')\nif ALLOW_TF32:\n",
        )
    notebook["cells"][10]["source"] = command_text.splitlines(keepends=True)

    notebook["cells"][-1]["source"] = markdown(
        """## Result

This run continues from the generic-prompt checkpoint-50 with lower LR `3e-6`,
linear decay over 25 continuation steps, validation/checkpoints every 5 steps,
and a separate output directory.
"""
    )["source"]

    target_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def write_compare_notebook() -> None:
    source_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Micro_Ablation_Checkpoints.ipynb"
    target_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Checkpoint50_Low_LR_Decay.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    notebook["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Compare BG / HAIR / FACE Checkpoint-50 Low-LR Decay

Use this notebook after the low-LR continuation run completes.
It compares base, the original generic checkpoint-50, and continuation checkpoints.
"""
    )["source"]

    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace(
            "MICRO_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'",
            f"MICRO_RUN_NAME = '{RUN_NAME}'",
        )
        text = text.replace(
            "COMPARISON_RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation_compare'",
            f"COMPARISON_RUN_NAME = '{COMPARE_NAME}'",
        )
        cell["source"] = text.splitlines(keepends=True)

    config_text = "".join(notebook["cells"][2]["source"])
    if "SOURCE_RUN_NAME =" not in config_text:
        config_text = config_text.replace(
            "MICRO_RUN_LOCAL = Path(f'/content/{MICRO_RUN_NAME}')\n",
            (
                "MICRO_RUN_LOCAL = Path(f'/content/{MICRO_RUN_NAME}')\n"
                f"SOURCE_RUN_NAME = '{SOURCE_RUN_NAME}'\n"
                "SOURCE_RUN_DRIVE = RUNS_DRIVE_ROOT / SOURCE_RUN_NAME\n"
                "SOURCE_RUN_LOCAL = Path(f'/content/{SOURCE_RUN_NAME}')\n"
            ),
        )
    variants_start = "MODEL_VARIANTS = [\n"
    variants_end = "]\nMAX_COMPARE_SAMPLES = 3\n"
    start = config_text.index(variants_start)
    end = config_text.index(variants_end) + 2
    variant_text = """MODEL_VARIANTS = [
    {'name': 'base_model', 'type': 'base'},
    {'name': 'generic_checkpoint_50', 'type': 'lora', 'run_dir': SOURCE_RUN_LOCAL, 'checkpoint_name': 'checkpoint-50'},
    {'name': 'continue_checkpoint_55', 'type': 'lora', 'run_dir': MICRO_RUN_LOCAL, 'checkpoint_name': 'checkpoint-55'},
    {'name': 'continue_checkpoint_60', 'type': 'lora', 'run_dir': MICRO_RUN_LOCAL, 'checkpoint_name': 'checkpoint-60'},
    {'name': 'continue_checkpoint_65', 'type': 'lora', 'run_dir': MICRO_RUN_LOCAL, 'checkpoint_name': 'checkpoint-65'},
    {'name': 'continue_checkpoint_70', 'type': 'lora', 'run_dir': MICRO_RUN_LOCAL, 'checkpoint_name': 'checkpoint-70'},
    {'name': 'continue_checkpoint_75', 'type': 'lora', 'run_dir': MICRO_RUN_LOCAL, 'checkpoint_name': 'checkpoint-75'},
]
"""
    config_text = config_text[:start] + variant_text + config_text[end:]
    notebook["cells"][2]["source"] = config_text.splitlines(keepends=True)

    copy_text = "".join(notebook["cells"][5]["source"])
    if "assert SOURCE_RUN_DRIVE.exists()" not in copy_text:
        copy_text = copy_text.replace(
            "assert MICRO_RUN_DRIVE.exists(), f'Missing mirrored micro-ablation alpha-weighted run on Drive: {MICRO_RUN_DRIVE}'\n",
            "assert MICRO_RUN_DRIVE.exists(), f'Missing continuation run on Drive: {MICRO_RUN_DRIVE}'\nassert SOURCE_RUN_DRIVE.exists(), f'Missing source run on Drive: {SOURCE_RUN_DRIVE}'\n",
        )
    if "if SOURCE_RUN_LOCAL.exists():" not in copy_text:
        copy_text = copy_text.replace(
            "if MICRO_RUN_LOCAL.exists():\n    shutil.rmtree(MICRO_RUN_LOCAL)\n",
            "if MICRO_RUN_LOCAL.exists():\n    shutil.rmtree(MICRO_RUN_LOCAL)\nif SOURCE_RUN_LOCAL.exists():\n    shutil.rmtree(SOURCE_RUN_LOCAL)\n",
        )
    if "shutil.copytree(SOURCE_RUN_DRIVE, SOURCE_RUN_LOCAL)" not in copy_text:
        copy_text = copy_text.replace(
            "shutil.copytree(MICRO_RUN_DRIVE, MICRO_RUN_LOCAL)\n",
            "shutil.copytree(MICRO_RUN_DRIVE, MICRO_RUN_LOCAL)\nshutil.copytree(SOURCE_RUN_DRIVE, SOURCE_RUN_LOCAL)\n",
        )
    notebook["cells"][5]["source"] = copy_text.splitlines(keepends=True)

    notebook["cells"][-1]["source"] = markdown(
        """## Result

Review the saved comparison directories on Drive to judge whether the low-LR
continuation improved over the original generic checkpoint-50 without drifting.
"""
    )["source"]

    target_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main() -> None:
    write_training_notebook()
    write_compare_notebook()
    print("Wrote low-LR continuation BG/HAIR/FACE notebooks.")


if __name__ == "__main__":
    main()
