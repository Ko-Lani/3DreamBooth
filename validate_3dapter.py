"""
Validation script for a pretrained 3dapter.
Loads a pretrained 3dapter and generates videos.

Usage:
  Single GPU:
        python validate_3dapter.py --tdapter_path <path> --reference_path <img> --prompts "prompt1" "prompt2"

  Multi-GPU (sequence parallel):
        torchrun --nproc_per_node=N validate_3dapter.py --tdapter_path <path> ...
"""

import os

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import datetime
import torch
import einops
import imageio
from loguru import logger
from torch import distributed as dist
from PIL import Image
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode, resize

from hyvideo.pipelines.tdapter_pipeline import TdapterPipeline
from hyvideo.commons.parallel_states import initialize_parallel_state


TDAPTER_INTERNAL_NAME = "tdapter"

# Initialize parallel state (works with single GPU too)
initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def is_rank0():
    return int(os.environ.get('RANK', '0')) == 0


def save_video(video: torch.Tensor, path: str):
    if video.ndim == 5:
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=24)


def encode_vae(pipe, images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 4:
        images = images.unsqueeze(2)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), pipe.vae.memory_efficient_context():
        latents = pipe.vae.encode(images).latent_dist.sample()
        if hasattr(pipe.vae.config, "shift_factor") and pipe.vae.config.shift_factor:
            latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        else:
            latents = latents * pipe.vae.config.scaling_factor
    return latents


def encode_reference(pipe, image_path: str, target_h=512, target_w=512) -> torch.Tensor:
    """Load, resize, center-crop, and VAE-encode a single reference image."""
    ref_image = Image.open(image_path).convert("RGB")
    ref_tensor = TT.functional.to_tensor(ref_image)  # [C, H, W]
    ref_tensor = (ref_tensor - 0.5) / 0.5  # [0,1] -> [-1,1]
    ref_tensor = ref_tensor.unsqueeze(0)  # [1, C, H, W]

    _, _, h, w = ref_tensor.shape
    if w / h > target_w / target_h:
        ref_tensor = resize(ref_tensor, [target_h, int(w * target_h / h)], interpolation=InterpolationMode.BICUBIC)
    else:
        ref_tensor = resize(ref_tensor, [int(h * target_w / w), target_w], interpolation=InterpolationMode.BICUBIC)
    _, _, h, w = ref_tensor.shape
    ref_tensor = TT.functional.crop(ref_tensor, (h - target_h) // 2, (w - target_w) // 2, target_h, target_w)
    ref_tensor = ref_tensor.clamp(-1.0, 1.0).to(device=pipe.vae.device, dtype=pipe.vae.dtype)

    return encode_vae(pipe, ref_tensor)


def load_pipeline(args):
    """Create pipeline and load LoRA adapter."""
    transformer_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32

    pipe = TdapterPipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=TdapterPipeline.get_transformer_version(
            args.resolution, 't2v', False, False, False
        ),
        create_sr_pipeline=False,
        transformer_dtype=transformer_dtype,
        flow_shift=args.flow_shift,
        device=torch.device('cuda'),
    )

    # Move everything to GPU
    device = torch.device("cuda")
    pipe.text_encoder.to(device)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)
    pipe.transformer.to(device)
    pipe.vae.to(device)

    # Load LoRA adapter
    if args.tdapter_path:
        logger.info(f"Loading 3dapter from {args.tdapter_path}")
        pipe.transformer.load_lora_adapter(
            pretrained_model_name_or_path_or_dict=args.tdapter_path,
            prefix=None,
            adapter_name=TDAPTER_INTERNAL_NAME,
            weight_name="pytorch_lora_weights.safetensors",
            hotswap=False,
        )
        pipe.transformer.set_adapter(TDAPTER_INTERNAL_NAME)
        logger.info("3dapter loaded successfully")

        # DEBUG: Print LoRA weight norms for comparison with training
        from peft.tuners.tuners_utils import BaseTunerLayer
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, BaseTunerLayer):
                for adapter in module.active_adapters:
                    if hasattr(module, 'lora_A') and adapter in module.lora_A:
                        a_norm = module.lora_A[adapter].weight.data.float().norm().item()
                        b_norm = module.lora_B[adapter].weight.data.float().norm().item()
                        logger.info(f"LORA_NORM {name} A={a_norm:.6f} B={b_norm:.6f}")
                break  # 첫 번째 모듈만

    return pipe


def validate(pipe, args):
    """Run validation: generate videos for each prompt."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Encode reference image(s)
    ref_latents = None
    if args.reference_path:
        with torch.no_grad():
            ref_latents = encode_reference(pipe, args.reference_path, target_h=720, target_w=1280)
            logger.info(f"Reference encoded: {args.reference_path} -> {ref_latents.shape}")

    for idx, prompt in enumerate(args.prompts):
        # Support "ref_path::prompt" format
        per_prompt_ref = ref_latents
        if "::" in prompt:
            left, right = prompt.split("::", 1)
            if os.path.isfile(left.strip()):
                with torch.no_grad():
                    per_prompt_ref = encode_reference(pipe, left.strip(), target_h=720, target_w=1280)
                prompt = right.strip()

        logger.info(f"[{idx+1}/{len(args.prompts)}] Generating: {prompt[:80]}...")

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                reference_latents=per_prompt_ref,
                aspect_ratio=args.aspect_ratio,
                video_length=args.video_length,
                enable_sr=False,
                prompt_rewrite=False,
                output_type="pt",
                seed=args.seed,
            )

        if is_rank0():
            video_path = os.path.join(args.output_dir, f"val_{idx:03d}.mp4")
            save_video(output.videos, video_path)
            logger.info(f"Saved: {video_path}")

    if is_rank0():
        logger.info(f"Validation complete. {len(args.prompts)} videos saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Validate a trained 3dapter")

    parser.add_argument("--model_path", type=str, default="ckpts",
                        help="Path to pretrained HunyuanVideo model")
    parser.add_argument("--tdapter_path", type=str, required=True,
                        help="Path to the 3dapter directory")
    parser.add_argument("--reference_path", type=str, default=None,
                        help="Path to reference image (used for all prompts unless per-prompt ref is given)")
    parser.add_argument("--prompts", type=str, nargs="+", required=True,
                        help="Validation prompts. Supports 'image_path::prompt text' format for per-prompt references.")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs/tdapter",
                        help="Output directory for generated videos")
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"])
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--video_length", type=int, default=121,
                        help="Number of frames (default: 121)")
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    args = parser.parse_args()

    pipe = load_pipeline(args)
    validate(pipe, args)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
