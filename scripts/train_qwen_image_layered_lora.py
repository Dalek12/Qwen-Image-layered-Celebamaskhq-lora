#!/usr/bin/env python
"""Colab-ready generic LoRA trainer for Qwen-Image-Layered."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import inspect
import json
import logging
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_celebmaskhq import create_generic_layered_dataloader


LOGGER = logging.getLogger("train_qwen_image_layered_lora")
DEFAULT_PROMPT = "decompose this portrait into editable portrait layers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-model-name-or-path", default="Qwen/Qwen-Image-Layered")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resolution", type=int, default=640, choices=[640, 1024])
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--validation-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", default="constant", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--lr-scheduler-steps", type=int, default=None)
    parser.add_argument("--lr-num-cycles", type=int, default=1)
    parser.add_argument("--lr-power", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=32)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--drop-warning-samples", action="store_true", default=True)
    parser.add_argument("--keep-warning-samples", action="store_true")
    parser.add_argument("--weighting-scheme", default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--validation-steps", type=int, default=200)
    parser.add_argument("--num-validation-batches", type=int, default=4)
    parser.add_argument("--checkpointing-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--mirror-output-dir", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--scheduler-shift", type=float, default=3.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.keep_warning_samples:
        args.drop_warning_samples = False
    return args


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_dataloader(args: argparse.Namespace, split: str, batch_size: int, max_samples: int | None, shuffle: bool):
    return create_generic_layered_dataloader(
        args.processed_root,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        resolution=args.resolution,
        drop_warning_samples=args.drop_warning_samples,
        max_samples=max_samples,
        max_layers=args.max_layers,
        pad_to_max_layers=args.max_layers,
    )


def resolve_weight_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[name]


def retrieve_latents(encoder_output: Any) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from VAE encoder output.")


def encode_qwen_latents(vae: Any, pixels: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    if pixels.dim() == 4:
        pixels = pixels.unsqueeze(2)
    latents = retrieve_latents(vae.encode(pixels.to(device=next(vae.parameters()).device, dtype=vae.dtype)))
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)
    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    return ((latents - mean) * inv_std).to(dtype=out_dtype)


def get_sigmas(scheduler: Any, timesteps: torch.Tensor, n_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sched_timesteps = scheduler.timesteps.to(device=device)
    sched_sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    indices = [(sched_timesteps == timestep).nonzero(as_tuple=False).item() for timestep in timesteps.to(device=device)]
    sigma = sched_sigmas[indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def encode_prompt_once(model_name: str, revision: str | None, prompt: str, weight_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    from diffusers import QwenImageLayeredPipeline
    from transformers import Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

    LOGGER.info("Encoding prompt once with the Qwen text stack.")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name, subfolder="tokenizer", revision=revision)
    processor = Qwen2VLProcessor.from_pretrained(model_name, subfolder="processor", revision=revision)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, subfolder="text_encoder", revision=revision, torch_dtype=weight_dtype)
    pipe = QwenImageLayeredPipeline.from_pretrained(
        model_name,
        revision=revision,
        tokenizer=tokenizer,
        processor=processor,
        text_encoder=text_encoder,
        vae=None,
        transformer=None,
        scheduler=None,
        torch_dtype=weight_dtype,
    )
    with torch.no_grad():
        prompt_embeds, prompt_mask = pipe.encode_prompt(prompt=prompt, max_sequence_length=512)
    del pipe, text_encoder, processor, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prompt_embeds = prompt_embeds.detach().cpu()
    if prompt_mask is not None:
        prompt_mask = prompt_mask.detach().cpu()
    return prompt_embeds, prompt_mask


def repeat_prompt_embeddings(prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    embeds = prompt_embeds.to(device=device, dtype=dtype).repeat(batch_size, 1, 1)
    if prompt_mask is None:
        return embeds, None
    return embeds, prompt_mask.to(device=device).repeat(batch_size, 1)


def call_transformer(transformer: Any, hidden_states: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor | None, img_shapes: Any, additional_t_cond: torch.Tensor | None) -> torch.Tensor:
    params = set(inspect.signature(transformer.forward).parameters.keys())
    kwargs: dict[str, Any] = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": prompt_embeds,
        "timestep": timestep / 1000,
        "return_dict": False,
    }
    if prompt_mask is not None and "encoder_hidden_states_mask" in params:
        kwargs["encoder_hidden_states_mask"] = prompt_mask
    if prompt_mask is not None and "txt_seq_lens" in params:
        kwargs["txt_seq_lens"] = prompt_mask.sum(dim=1).tolist()
    if additional_t_cond is not None and "additional_t_cond" in params:
        kwargs["additional_t_cond"] = additional_t_cond
    if "guidance" in params and getattr(transformer.config, "guidance_embeds", False):
        kwargs["guidance"] = torch.ones((hidden_states.shape[0],), device=hidden_states.device, dtype=torch.float32)
    if "img_shapes" in params:
        errors: list[str] = []
        candidates: list[tuple[str, Any]] = [("raw", img_shapes)]
        if img_shapes and isinstance(img_shapes[0], tuple):
            candidates.append(("nested_single", [[shape] for shape in img_shapes]))
        for name, candidate in candidates:
            try:
                kwargs["img_shapes"] = candidate
                return transformer(**kwargs)[0]
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
        raise RuntimeError("Transformer call failed for all img_shapes variants. " + " | ".join(errors))
    return transformer(**kwargs)[0]


def save_lora(accelerator: Any, transformer: Any, pipeline_cls: Any, output_dir: Path, step: int) -> None:
    from peft.utils import get_peft_model_state_dict

    model = accelerator.unwrap_model(transformer)
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    ckpt = output_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    pipeline_cls.save_lora_weights(ckpt, transformer_lora_layers=get_peft_model_state_dict(model))


def parse_checkpoint_step(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.split("-", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None


def list_checkpoint_dirs(output_dir: Path) -> list[Path]:
    checkpoint_dirs: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        step = parse_checkpoint_step(path)
        if step is None:
            continue
        checkpoint_dirs.append((step, path))
    checkpoint_dirs.sort(key=lambda item: item[0])
    return [path for _, path in checkpoint_dirs]


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = list_checkpoint_dirs(output_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def resolve_resume_checkpoint(output_dir: Path, resume_arg: str | None) -> Path | None:
    if not resume_arg:
        return None
    if resume_arg == "latest":
        checkpoint_dir = find_latest_checkpoint(output_dir)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No checkpoint-* directories found under {output_dir} for --resume-from-checkpoint latest")
        return checkpoint_dir
    checkpoint_dir = Path(resume_arg).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint_training_state(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "training_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint training state: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def sync_output_dir_to_mirror(output_dir: Path, mirror_output_dir: Path | None) -> None:
    if mirror_output_dir is None:
        return
    mirror_output_dir = mirror_output_dir.resolve()
    mirror_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if mirror_output_dir.exists():
        shutil.rmtree(mirror_output_dir)
    shutil.copytree(output_dir, mirror_output_dir)


def save_checkpoint(
    accelerator: Any,
    transformer: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    pipeline_cls: Any,
    output_dir: Path,
    mirror_output_dir: Path | None,
    step: int,
    epoch: int,
    batch_index: int,
    completed_batches: int,
    latest_learning_rate: float,
    keep: int,
) -> Path:
    save_lora(accelerator, transformer, pipeline_cls, output_dir, step)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), checkpoint_dir / "lr_scheduler.pt")
    training_state = {
        "global_step": step,
        "epoch": epoch,
        "batch_index": batch_index,
        "completed_batches": completed_batches,
        "latest_learning_rate": latest_learning_rate,
    }
    write_json(checkpoint_dir / "training_state.json", training_state)
    write_json(output_dir / "trainer_state.json", {**training_state, "latest_checkpoint": checkpoint_dir.name})
    cleanup_checkpoints(output_dir, keep)
    sync_output_dir_to_mirror(output_dir, mirror_output_dir)
    return checkpoint_dir


def cleanup_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = list_checkpoint_dirs(output_dir)
    while len(checkpoints) > keep:
        shutil.rmtree(checkpoints.pop(0), ignore_errors=True)


def load_resume_lora_into_transformer(pipeline_cls: Any, transformer: Any, checkpoint_dir: Path, adapter_name: str = "default") -> None:
    lora_state = pipeline_cls.lora_state_dict(checkpoint_dir, weight_name="pytorch_lora_weights.safetensors")
    if isinstance(lora_state, tuple):
        state_dict = lora_state[0]
        metadata = lora_state[2] if len(lora_state) > 2 else None
    else:
        state_dict = lora_state
        metadata = None
    pipeline_cls.load_lora_into_transformer(
        state_dict,
        transformer=transformer,
        adapter_name=adapter_name,
        _pipeline=None,
        hotswap=True,
        metadata=metadata,
    )


def ensure_lora_adapter_trainable(transformer: Any, adapter_name: str = "default") -> None:
    """Re-activate the LoRA adapter for training after creation or resume loads."""

    if hasattr(transformer, "set_adapter"):
        set_adapter_params = inspect.signature(transformer.set_adapter).parameters
        if "inference_mode" in set_adapter_params:
            transformer.set_adapter(adapter_name, inference_mode=False)
        else:
            transformer.set_adapter(adapter_name)
    if hasattr(transformer, "set_requires_grad"):
        transformer.set_requires_grad([adapter_name], requires_grad=True)

    # Some PEFT/diffusers combinations leave the active adapter frozen after hotswap loads.
    # Fall back to toggling LoRA parameters directly so optimizer construction never sees an empty set.
    for name, param in transformer.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)


def build_latent_layer_valid_mask(layer_valid_mask: torch.Tensor, latent_frame_count: int, dtype: torch.dtype) -> torch.Tensor:
    """Downsample the padded frame-valid mask to the latent temporal resolution."""

    mask_5d = layer_valid_mask[:, None, :, None, None].float()
    pooled = F.adaptive_max_pool3d(mask_5d, output_size=(latent_frame_count, 1, 1))
    return pooled.to(dtype=dtype)


def build_latent_alpha_loss_weight(
    target_rgba: torch.Tensor,
    latent_shape: torch.Size,
    latent_layer_valid_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Approximate per-latent loss weight from the visible alpha coverage of each target layer."""

    alpha = target_rgba[:, 3:4].float()
    pooled_alpha = F.adaptive_max_pool3d(
        alpha,
        output_size=(latent_shape[2], latent_shape[3], latent_shape[4]),
    )
    # Keep a small floor so sparse layers still preserve some context while emphasizing visible pixels.
    alpha_weight = (0.05 + 0.95 * pooled_alpha).to(dtype=dtype)
    return alpha_weight * latent_layer_valid_mask


def compute_batch_loss(batch: dict[str, Any], accelerator: Any, vae: Any, transformer: Any, scheduler: Any, pipeline_cls: Any, prompt_embeds: torch.Tensor, prompt_mask: torch.Tensor | None, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

    device = accelerator.device
    target_rgba = batch["target_rgba_layers"].to(device=device, dtype=vae.dtype).permute(0, 2, 1, 3, 4).contiguous()
    conditioning_rgba = batch["conditioning_rgba"].to(device=device, dtype=vae.dtype)
    layer_valid_mask = batch["layer_valid_mask"].to(device=device, dtype=vae.dtype)

    with torch.no_grad():
        target_latents = encode_qwen_latents(vae, target_rgba, transformer.dtype)
        condition_latents = encode_qwen_latents(vae, conditioning_rgba, transformer.dtype)
    latent_layer_valid_mask = build_latent_layer_valid_mask(
        layer_valid_mask,
        latent_frame_count=target_latents.shape[2],
        dtype=target_latents.dtype,
    )
    latent_alpha_loss_weight = build_latent_alpha_loss_weight(
        target_rgba=target_rgba,
        latent_shape=target_latents.shape,
        latent_layer_valid_mask=latent_layer_valid_mask,
        dtype=target_latents.dtype,
    )
    target_latents = target_latents * latent_layer_valid_mask

    noise = torch.randn_like(target_latents) * latent_layer_valid_mask
    batch_size = target_latents.shape[0]
    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=batch_size,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )
    indices = (u * scheduler.config.num_train_timesteps).long().clamp(max=scheduler.config.num_train_timesteps - 1)
    timesteps = scheduler.timesteps[indices].to(device=device)
    sigmas = get_sigmas(scheduler, timesteps, target_latents.ndim, device, target_latents.dtype)
    noisy_target = ((1.0 - sigmas) * target_latents + sigmas * noise) * latent_layer_valid_mask

    packed_target = pipeline_cls._pack_latents(
        noisy_target.permute(0, 2, 1, 3, 4).contiguous(),
        batch_size=batch_size,
        num_channels_latents=target_latents.shape[1],
        height=target_latents.shape[3],
        width=target_latents.shape[4],
        layers=target_latents.shape[2],
    )
    packed_condition = pipeline_cls._pack_latents(
        condition_latents.permute(0, 2, 1, 3, 4).contiguous(),
        batch_size=batch_size,
        num_channels_latents=condition_latents.shape[1],
        height=condition_latents.shape[3],
        width=condition_latents.shape[4],
        layers=condition_latents.shape[2],
    )
    hidden_states = torch.cat([packed_target, packed_condition], dim=1)

    batch_prompt_embeds, batch_prompt_mask = repeat_prompt_embeddings(prompt_embeds, prompt_mask, batch_size, device, transformer.dtype)
    latent_patch_height = target_latents.shape[3] // 2
    latent_patch_width = target_latents.shape[4] // 2
    conditioning_patch_height = condition_latents.shape[3] // 2
    conditioning_patch_width = condition_latents.shape[4] // 2
    img_shapes = [
        [
            *[(1, latent_patch_height, latent_patch_width) for _ in range(target_latents.shape[2])],
            (1, conditioning_patch_height, conditioning_patch_width),
        ]
        for _ in range(batch_size)
    ]
    additional_t_cond = torch.zeros((batch_size,), device=device, dtype=torch.long)

    model_pred = call_transformer(
        transformer,
        hidden_states=hidden_states,
        timestep=timesteps.expand(batch_size).to(dtype=transformer.dtype),
        prompt_embeds=batch_prompt_embeds,
        prompt_mask=batch_prompt_mask,
        img_shapes=img_shapes,
        additional_t_cond=additional_t_cond,
    )
    model_pred = model_pred[:, : packed_target.shape[1]]
    model_pred = pipeline_cls._unpack_latents(
        model_pred,
        target_latents.shape[3] * (2 ** len(vae.temperal_downsample)),
        target_latents.shape[4] * (2 ** len(vae.temperal_downsample)),
        max(target_latents.shape[2] - 1, 0),
        2 ** len(vae.temperal_downsample),
    )
    if model_pred.dim() == 4:
        model_pred = model_pred.unsqueeze(2)

    target = (noise - target_latents) * latent_layer_valid_mask
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    sq_error = ((model_pred.float() - target.float()) ** 2) * latent_alpha_loss_weight.float()
    denom = latent_alpha_loss_weight.float().sum().clamp_min(1.0)
    mse = sq_error.sum() / denom
    loss = torch.mean((weighting.float() * sq_error).reshape(batch_size, -1), dim=1).mean()
    return loss, mse, batch_size


def main() -> None:
    setup_logging()
    args = parse_args()

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import AutoencoderKLQwenImage, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler, QwenImageLayeredPipeline, QwenImageTransformer2DModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, prepare_model_for_kbit_training

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir).resolve()
    mirror_output_dir = None if args.mirror_output_dir is None else Path(args.mirror_output_dir).resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logs_dir)),
    )
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = resolve_weight_dtype(args.mixed_precision)
    resume_checkpoint = resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
    run_config_payload = {**vars(args), "utc_started_at": datetime.now(timezone.utc).isoformat()}
    if resume_checkpoint is not None:
        run_config_payload["resume_checkpoint"] = str(resume_checkpoint)
    write_json(output_dir / "run_config.json", run_config_payload)
    train_loader = build_dataloader(args, args.train_split, args.train_batch_size, args.max_train_samples, True)
    val_loader = None if args.skip_validation else build_dataloader(args, args.validation_split, args.validation_batch_size, args.max_validation_samples, False)
    prompt_embeds, prompt_mask = encode_prompt_once(args.pretrained_model_name_or_path, args.revision, args.prompt, weight_dtype)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision, shift=args.scheduler_shift)
    scheduler_train = copy.deepcopy(scheduler)
    vae = AutoencoderKLQwenImage.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=args.bnb_4bit_quant_type, bnb_4bit_compute_dtype=weight_dtype)
    transformer = QwenImageTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, quantization_config=quant_config, torch_dtype=weight_dtype)
    if args.load_in_4bit:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    transformer.requires_grad_(False)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=[name.strip() for name in args.lora_target_modules.split(",") if name.strip()],
        ),
        adapter_name="default",
    )
    if resume_checkpoint is not None:
        LOGGER.info("Loading LoRA weights from checkpoint: %s", resume_checkpoint)
        load_resume_lora_into_transformer(QwenImageLayeredPipeline, transformer, resume_checkpoint, adapter_name="default")
    ensure_lora_adapter_trainable(transformer, adapter_name="default")
    trainable_params = [param for param in transformer.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable LoRA parameters were found after adapter setup. "
            "This usually means the active adapter stayed frozen in the current Colab runtime copy."
        )
    for param in trainable_params:
        if param.dtype != torch.float32:
            param.data = param.data.float()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_steps = min(args.max_train_steps, args.num_train_epochs * steps_per_epoch)
    lr_scheduler_steps = args.lr_scheduler_steps if args.lr_scheduler_steps is not None else max_steps
    if lr_scheduler_steps <= 0:
        raise ValueError("--lr-scheduler-steps must be positive when provided.")
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=lr_scheduler_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_loader, lr_scheduler)

    start_global_step = 0
    start_epoch = 0
    resume_batches_to_skip = 0
    if resume_checkpoint is not None:
        training_state = load_checkpoint_training_state(resume_checkpoint)
        optimizer_state_path = resume_checkpoint / "optimizer.pt"
        lr_scheduler_state_path = resume_checkpoint / "lr_scheduler.pt"
        if optimizer_state_path.is_file() and not args.reset_optimizer_on_resume:
            optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
        elif args.reset_optimizer_on_resume:
            LOGGER.info("Resetting optimizer state on resume; LoRA weights are still loaded from checkpoint.")
        if lr_scheduler_state_path.is_file() and not args.reset_optimizer_on_resume:
            lr_scheduler.load_state_dict(torch.load(lr_scheduler_state_path, map_location="cpu"))
        elif args.reset_optimizer_on_resume:
            LOGGER.info("Resetting LR scheduler state on resume; using %s schedule over %s scheduler steps.", args.lr_scheduler, lr_scheduler_steps)
        start_global_step = int(training_state.get("global_step", 0))
        completed_batches = int(training_state.get("completed_batches", start_global_step * args.gradient_accumulation_steps))
        train_loader_length = max(len(train_loader), 1)
        start_epoch = completed_batches // train_loader_length
        resume_batches_to_skip = completed_batches % train_loader_length
        LOGGER.info(
            "Resuming training from step %s | epoch %s | skipping %s batches in the first resumed epoch",
            start_global_step,
            start_epoch,
            resume_batches_to_skip,
        )

    LOGGER.info("Training samples: %s | validation enabled: %s | max steps: %s", len(train_loader.dataset), val_loader is not None, max_steps)
    progress = tqdm(total=max_steps, initial=start_global_step, disable=not accelerator.is_local_main_process)
    log_path = logs_dir / "training_metrics.jsonl"
    global_step = start_global_step
    last_epoch = start_epoch
    last_batch_index = -1

    for epoch in range(start_epoch, args.num_train_epochs):
        last_epoch = epoch
        transformer.train()
        for batch_index, batch in enumerate(train_loader):
            last_batch_index = batch_index
            if epoch == start_epoch and batch_index < resume_batches_to_skip:
                continue
            with accelerator.accumulate(transformer):
                loss, mse, batch_size = compute_batch_loss(batch, accelerator, vae, transformer, scheduler_train, QwenImageLayeredPipeline, prompt_embeds, prompt_mask, args)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            train_row = {"step": global_step, "epoch": epoch, "train_loss": round(float(loss.detach().cpu().item()), 8), "train_mse_loss": round(float(mse.detach().cpu().item()), 8), "learning_rate": round(float(lr_scheduler.get_last_lr()[0]), 12), "batch_size": int(batch_size)}
            if accelerator.is_main_process:
                append_jsonl(log_path, train_row)
            progress.set_postfix({"loss": f"{train_row['train_loss']:.5f}"})

            if val_loader is not None and args.validation_steps > 0 and global_step % args.validation_steps == 0:
                transformer.eval()
                val_losses, val_mses = [], []
                with torch.no_grad():
                    for index, val_batch in enumerate(val_loader):
                        if index >= args.num_validation_batches:
                            break
                        val_loss, val_mse, _ = compute_batch_loss(val_batch, accelerator, vae, transformer, scheduler_train, QwenImageLayeredPipeline, prompt_embeds, prompt_mask, args)
                        val_losses.append(float(val_loss.detach().cpu().item()))
                        val_mses.append(float(val_mse.detach().cpu().item()))
                transformer.train()
                if val_losses and accelerator.is_main_process:
                    append_jsonl(log_path, {"step": global_step, "epoch": epoch, "validation_loss": round(sum(val_losses) / len(val_losses), 8), "validation_mse_loss": round(sum(val_mses) / len(val_mses), 8), "num_validation_batches": len(val_losses)})

            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                completed_batches = epoch * len(train_loader) + batch_index + 1
                save_checkpoint(
                    accelerator=accelerator,
                    transformer=transformer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    pipeline_cls=QwenImageLayeredPipeline,
                    output_dir=output_dir,
                    mirror_output_dir=mirror_output_dir,
                    step=global_step,
                    epoch=epoch,
                    batch_index=batch_index,
                    completed_batches=completed_batches,
                    latest_learning_rate=lr_scheduler.get_last_lr()[0],
                    keep=args.save_total_limit,
                )

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        train_loader_length = max(len(train_loader), 1)
        if global_step > 0:
            completed_batches = last_epoch * train_loader_length + max(last_batch_index, 0) + 1
        else:
            completed_batches = 0
        final_checkpoint_dir = save_checkpoint(
            accelerator=accelerator,
            transformer=transformer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            pipeline_cls=QwenImageLayeredPipeline,
            output_dir=output_dir,
            mirror_output_dir=mirror_output_dir,
            step=global_step,
            epoch=last_epoch,
            batch_index=last_batch_index,
            completed_batches=completed_batches,
            latest_learning_rate=lr_scheduler.get_last_lr()[0],
            keep=args.save_total_limit,
        )
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        write_json(final_dir / "final_summary.json", {"global_step": global_step, "max_steps": max_steps, "prompt": args.prompt, "resolution": args.resolution, "max_layers": args.max_layers, "train_split": args.train_split, "validation_split": None if args.skip_validation else args.validation_split, "latest_checkpoint": final_checkpoint_dir.name})
        sync_output_dir_to_mirror(output_dir, mirror_output_dir)
    accelerator.end_training()
    LOGGER.info("Finished training at step %s", global_step)


if __name__ == "__main__":
    main()
