"""Create prompt-aligned Colab notebooks for the BG/HAIR/FACE 5k experiment."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
PROMPT = "decompose this portrait into exactly three editable RGBA layers: background, hair, and face"
RUN_NAME = "qwen_layered_lora_bghairface_5k_prompt_aligned_micro_ablation"
COMPARE_NAME = "qwen_layered_lora_bghairface_5k_prompt_aligned_micro_ablation_compare"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def write_training_notebook() -> None:
    source_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Micro_Ablation.ipynb"
    target_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Training_BG_HAIR_FACE_5K_Prompt_Aligned_Micro_Ablation.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    notebook["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Training BG / HAIR / FACE 5k Prompt-Aligned Micro Ablation

Use this notebook with a GPU runtime after the BG/HAIR/FACE CPU notebook creates the train/val shard package.
This run keeps the same weak alpha-weighted settings, but changes the text prompt to name the exact target layers.
"""
    )["source"]

    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace(
            "RUN_NAME = 'qwen_layered_lora_bghairface_5k_micro_ablation'",
            f"RUN_NAME = '{RUN_NAME}'",
        )
        text = text.replace(
            "print('Micro ablation defaults: lr=1e-5, rank=8, alpha=8, grad_accum=4, max_steps=100, checkpoint_every=25')",
            "print('Prompt-aligned defaults: lr=1e-5, rank=8, alpha=8, grad_accum=4, max_steps=100, checkpoint_every=25')",
        )
        text = text.replace(
            "print('This alpha-weighted run uses a fresh RUN_NAME so prior micro-ablation checkpoints are not reused.')",
            "print('This run uses an explicit BG/HAIR/FACE prompt and a fresh RUN_NAME.')",
        )
        text = text.replace(
            "This run trains the 5k `BG / HAIR / FACE` target scheme with LR `1e-5`, rank `8`,\nLoRA alpha `8`, 100 steps, and checkpoints every 25 steps.",
            "This run trains the 5k `BG / HAIR / FACE` target scheme with an explicit layer prompt, LR `1e-5`, rank `8`, LoRA alpha `8`, 100 steps, and checkpoints every 25 steps.",
        )
        cell["source"] = text.splitlines(keepends=True)

    config_text = "".join(notebook["cells"][2]["source"])
    if "PROMPT =" not in config_text:
        config_text = config_text.replace(
            "MIXED_PRECISION = 'bf16'\n",
            f"MIXED_PRECISION = 'bf16'\nPROMPT = {PROMPT!r}\n",
        )
    if "print('Prompt:', PROMPT)" not in config_text:
        config_text = config_text.replace(
            "print('This run uses an explicit BG/HAIR/FACE prompt and a fresh RUN_NAME.')\n",
            "print('This run uses an explicit BG/HAIR/FACE prompt and a fresh RUN_NAME.')\nprint('Prompt:', PROMPT)\n",
        )
    notebook["cells"][2]["source"] = config_text.splitlines(keepends=True)

    command_text = "".join(notebook["cells"][10]["source"])
    if "'--prompt', PROMPT," not in command_text:
        command_text = command_text.replace(
            "    '--processed-root', str(PROCESSED_LOCAL),\n",
            "    '--processed-root', str(PROCESSED_LOCAL),\n    '--prompt', PROMPT,\n",
        )
    notebook["cells"][10]["source"] = command_text.splitlines(keepends=True)

    target_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def write_compare_notebook() -> None:
    source_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Micro_Ablation_Checkpoints.ipynb"
    target_path = NOTEBOOK_DIR / "CelebAMaskHQ_GPU_Compare_BG_HAIR_FACE_5K_Prompt_Aligned_Micro_Ablation_Checkpoints.ipynb"
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    notebook["cells"][0]["source"] = markdown(
        """# CelebAMask-HQ GPU Compare BG / HAIR / FACE 5k Prompt-Aligned Checkpoints

Use this notebook with a GPU runtime after the prompt-aligned BG/HAIR/FACE 5k micro-ablation run completes.
It compares the base model and prompt-aligned checkpoints with the same explicit layer prompt.
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
        text = text.replace(
            "Use this notebook with a GPU runtime after the BG/HAIR/FACE 5k micro-ablation run completes.",
            "Use this notebook with a GPU runtime after the prompt-aligned BG/HAIR/FACE 5k micro-ablation run completes.",
        )
        cell["source"] = text.splitlines(keepends=True)

    config_text = "".join(notebook["cells"][2]["source"])
    if "PROMPT =" not in config_text:
        config_text = config_text.replace(
            "FORCE_REBUILD_COMPARISON = True\n",
            f"FORCE_REBUILD_COMPARISON = True\nPROMPT = {PROMPT!r}\n",
        )
    if "print('Prompt:', PROMPT)" not in config_text:
        config_text = config_text.replace(
            "print('Comparison output on Drive:', COMPARISON_DRIVE)\n",
            "print('Comparison output on Drive:', COMPARISON_DRIVE)\nprint('Prompt:', PROMPT)\n",
        )
    notebook["cells"][2]["source"] = config_text.splitlines(keepends=True)

    infer_text = "".join(notebook["cells"][7]["source"])
    if "prompt=PROMPT," not in infer_text:
        infer_text = infer_text.replace(
            "                image=input_image,\n",
            "                image=input_image,\n                prompt=PROMPT,\n",
        )
    if "'prompt': PROMPT," not in infer_text:
        infer_text = infer_text.replace(
            "                'checkpoint_name': variant.get('checkpoint_name'),\n",
            "                'checkpoint_name': variant.get('checkpoint_name'),\n                'prompt': PROMPT,\n",
        )
        infer_text = infer_text.replace(
            "            'checkpoint_name': variant.get('checkpoint_name'),\n",
            "            'checkpoint_name': variant.get('checkpoint_name'),\n            'prompt': PROMPT,\n",
        )
    notebook["cells"][7]["source"] = infer_text.splitlines(keepends=True)

    notebook["cells"][-1]["source"] = markdown(
        """## Result

Review the saved comparison directories on Drive to judge whether the explicit
`background, hair, and face` prompt improves slot binding over the generic-prompt run.
"""
    )["source"]

    target_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def main() -> None:
    write_training_notebook()
    write_compare_notebook()
    print("Wrote prompt-aligned BG/HAIR/FACE notebooks.")


if __name__ == "__main__":
    main()
