#!/usr/bin/env python3
"""
Joint Checkpoint Inference Script

Load the subject adapter and 3dapter from a joint checkpoint and generate a video.
"""

import os
import argparse
import torch
import imageio
import einops
from pathlib import Path
import torch.distributed as dist

from hyvideo.pipelines.joint_pipeline import JointPipeline
from hyvideo.commons.parallel_states import initialize_parallel_state, get_parallel_state


TDAPTER_INTERNAL_NAME = "tdapter"
TDAPTER_EXPORT_DIR = "tdapter"
SUBJECT_ADAPTER_INTERNAL_NAME = "tdb"
SUBJECT_ADAPTER_EXPORT_DIR = "subject_adapter"


def save_video(video: torch.Tensor, path: str, fps: int = 24):
    """Save a generated video tensor as an MP4 file."""
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]

    # [C, F, H, W] -> [F, H, W, C]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid.cpu().numpy(), fps=fps)
    print(f"✅ Video saved to: {path}")


def load_reference_latents(pipeline, reference_path: str, target_h=720, target_w=1280):
    """
    Encode reference images with the VAE and return their latent representations.
    """
    import torchvision.transforms as TT
    from torchvision.transforms.functional import InterpolationMode, resize
    from PIL import Image

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    if not os.path.isdir(reference_path):
        raise ValueError(f"{reference_path} is not a directory!")

    img_files = [f for f in os.listdir(reference_path) if f.lower().endswith(valid_extensions)]
    img_files.sort()

    if not img_files:
        print(f"⚠️ No images found in {reference_path}")
        return None

    all_latents = []

    for img_name in img_files:
        full_path = os.path.join(reference_path, img_name)

        try:
            # Load and preprocess image
            ref_image = Image.open(full_path).convert("RGB")
            ref_tensor = TT.functional.to_tensor(ref_image)
            ref_tensor = (ref_tensor - 0.5) / 0.5  # [0, 1] -> [-1, 1]
            ref_tensor = ref_tensor.unsqueeze(0)

            # Resize and crop
            _, _, h, w = ref_tensor.shape
            if w / h > target_w / target_h:
                ref_tensor = resize(ref_tensor, [target_h, int(w * target_h / h)], interpolation=InterpolationMode.BICUBIC)
            else:
                ref_tensor = resize(ref_tensor, [int(h * target_w / w), target_w], interpolation=InterpolationMode.BICUBIC)

            _, _, h, w = ref_tensor.shape
            ref_tensor = TT.functional.crop(ref_tensor, (h - target_h) // 2, (w - target_w) // 2, target_h, target_w)
            ref_tensor = ref_tensor.clamp(-1.0, 1.0)

            # Convert to device and dtype
            ref_tensor = ref_tensor.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
            if len(ref_tensor.shape) == 4:
                ref_tensor = ref_tensor.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

            # VAE encode
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16), pipeline.vae.memory_efficient_context():
                enc_output = pipeline.vae.encode(ref_tensor)

                if hasattr(enc_output, "latent_dist"):
                    latents = enc_output.latent_dist.mode()
                elif hasattr(enc_output, "latent"):
                    latents = enc_output.latent
                else:
                    latents = enc_output

                if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor:
                    latents = (latents - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
                else:
                    latents = latents * pipeline.vae.config.scaling_factor

            all_latents.append(latents)
            print(f"✅ Encoded reference: {img_name}")

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

    return all_latents


def main():
    parser = argparse.ArgumentParser(description="Generate videos using a joint checkpoint")

    # Distributed parameters
    parser.add_argument(
        "--sp_size",
        type=int,
        default=1,
        help="Sequence parallelism size (default: 1 for single GPU, 4 for multi-GPU)"
    )

    # Model paths
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., ./outputs/joint/example/checkpoint-400)"
    )
    parser.add_argument(
        "--pretrained_model_root",
        type=str,
        default="./ckpts",
        help="Path to pretrained model root"
    )
    parser.add_argument(
        "--pretrained_transformer_version",
        type=str,
        default="720p_t2v",
        help="Transformer version (e.g., 720p_t2v, 480p_t2v)"
    )

    # LoRA adapter paths (optional, overrides checkpoint_path/lora/*)
    parser.add_argument(
        "--subject_adapter_path",
        type=str,
        default=None,
        help="Path to the subject adapter directory"
    )
    parser.add_argument(
        "--tdapter_path",
        type=str,
        default=None,
        help="Path to the 3dapter directory"
    )

    # Reference images
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to reference images folder"
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt (optional)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=129,
        help="Number of frames to generate (must be 4n+1, e.g., 129, 81, 49)"
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="Aspect ratio (e.g., 16:9, 9:16, 1:1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Flow shift parameter"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--text_lora_spans",
        type=str,
        nargs="+",
        default=None,
        help="Restrict text LoRA to specific token spans, e.g. --text_lora_spans 'rhs figure' 'kad floor'. "
             "Omit to apply LoRA to all text tokens."
    )

    args = parser.parse_args()

    # Initialize distributed environment
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        is_main_process = rank == 0
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_main_process = True

    # Initialize parallel state for sequence parallelism
    if world_size > 1:
        initialize_parallel_state(sp=args.sp_size, dp_replicate=1)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    def find_adapter(base_path, candidate_names, override_path=None):
        if override_path:
            p = Path(override_path)
            if p.exists():
                return p
        for name in candidate_names:
            for candidate in [base_path / "lora" / name, base_path / name]:
                if candidate.exists():
                    return candidate
        searched = ", ".join(candidate_names)
        raise ValueError(f"Adapter not found. Tried names [{searched}] under {base_path / 'lora'} and {base_path}")

    subject_adapter_path = find_adapter(
        checkpoint_path,
        [SUBJECT_ADAPTER_EXPORT_DIR, SUBJECT_ADAPTER_INTERNAL_NAME],
        args.subject_adapter_path,
    )
    tdapter_path = find_adapter(
        checkpoint_path,
        [TDAPTER_EXPORT_DIR, TDAPTER_INTERNAL_NAME],
        args.tdapter_path,
    )

    if is_main_process:
        print("=" * 80)
        print("🎬 Joint Checkpoint Inference")
        print("=" * 80)
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"📁 Subject adapter: {subject_adapter_path}")
        print(f"📁 3dapter: {tdapter_path}")
        print(f"📁 Reference images: {args.reference_path}")
        print(f"📝 Prompt: {args.prompt}")
        print(f"🎯 Video length: {args.video_length} frames")
        print(f"🎨 Aspect ratio: {args.aspect_ratio}")
        print(f"🌱 Seed: {args.seed}")
        if world_size > 1:
            print(f"🚀 Distributed: {world_size} GPUs, SP size: {args.sp_size}")
        print("=" * 80)

    # Create pipeline
    if is_main_process:
        print("\n🔧 Loading pipeline...")

    pipeline = JointPipeline.create_pipeline(
        pretrained_model_name_or_path=args.pretrained_model_root,
        transformer_version=args.pretrained_transformer_version,
        transformer_dtype=torch.bfloat16,
        enable_offloading=False,
        enable_group_offloading=False,
        overlap_group_offloading=False,
        create_sr_pipeline=False,
        flow_shift=args.flow_shift,
        device=device,
    )
    if is_main_process:
        print("✅ Pipeline loaded")

    # Load LoRA adapters
    if is_main_process:
        print("\n🔧 Loading LoRA adapters...")

    # Load 3dapter.
    if is_main_process:
        print(f"  Loading 3dapter from {tdapter_path}...")
    pipeline.transformer.load_lora_adapter(
        pretrained_model_name_or_path_or_dict=str(tdapter_path),
        prefix=None,
        adapter_name=TDAPTER_INTERNAL_NAME,
        use_safetensors=True,
        hotswap=False,
    )

    # Load subject adapter.
    if is_main_process:
        print(f"  Loading subject adapter from {subject_adapter_path}...")
    pipeline.transformer.load_lora_adapter(
        pretrained_model_name_or_path_or_dict=str(subject_adapter_path),
        prefix=None,
        adapter_name=SUBJECT_ADAPTER_INTERNAL_NAME,
        use_safetensors=True,
        hotswap=False,
    )

    # Activate both adapters
    from peft.tuners.tuners_utils import BaseTunerLayer
    for module in pipeline.transformer.modules():
        if isinstance(module, BaseTunerLayer):
            module.set_adapter([TDAPTER_INTERNAL_NAME, SUBJECT_ADAPTER_INTERNAL_NAME])

    if is_main_process:
        print("✅ Adapters loaded and activated: ['tdapter', 'subject_adapter']")

    # Load reference latents
    if is_main_process:
        print("\n🖼️ Encoding reference images...")
    ref_latents_list = load_reference_latents(
        pipeline,
        args.reference_path,
        target_h=720,
        target_w=1280
    )

    if ref_latents_list is None or len(ref_latents_list) == 0:
        raise ValueError("No reference latents loaded!")

    if is_main_process:
        print(f"✅ Encoded {len(ref_latents_list)} reference images")

    # Generate video
    if is_main_process:
        print("\n🎬 Generating video...")
        print(f"  Prompt: {args.prompt}")
        print(f"  Steps: {args.num_inference_steps}")
        print(f"  Guidance scale: {args.guidance_scale}")

    # Build span mask hook if requested
    from contextlib import contextmanager

    @contextmanager
    def inject_span_mask(prompt, spans):
        """Inject txt_span_mask into transformer.forward() via a pre-hook.

        Handles CFG batches ([uncond, cond]) by zeroing the unconditional rows.
        """
        if not spans:
            yield
            return

        tokenizer = pipeline.text_encoder.tokenizer

        def _compute_mask(prompt_text, seq_len):
            mask = torch.zeros(1, seq_len, dtype=torch.float32)
            try:
                enc = tokenizer(
                    prompt_text,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                    return_tensors=None,
                )
                offsets = enc["offset_mapping"]
            except Exception:
                mask[:] = 1.0
                return mask
            for span in spans:
                char_start = prompt_text.find(span)
                if char_start == -1:
                    continue
                char_end = char_start + len(span)
                for tok_idx, (ts, te) in enumerate(offsets):
                    if ts == te or te <= char_start or ts >= char_end:
                        continue
                    if 0 <= tok_idx < seq_len:
                        mask[0, tok_idx] = 1.0
            return mask

        def _pre_hook(module, args, kwargs):
            text_states = kwargs.get('text_states')
            if text_states is None:
                return args, kwargs
            B, L = text_states.shape[0], text_states.shape[1]
            single_mask = _compute_mask(prompt, L).to(device=text_states.device)
            if B > 1:
                # CFG: [uncond, cond] — zeros for uncond rows
                zeros = torch.zeros(B - 1, L, device=single_mask.device, dtype=single_mask.dtype)
                span_mask = torch.cat([zeros, single_mask], dim=0)
            else:
                span_mask = single_mask
            kwargs['txt_span_mask'] = span_mask
            return args, kwargs

        hook = pipeline.transformer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        try:
            yield
        finally:
            hook.remove()

    with torch.no_grad(), inject_span_mask(args.prompt, args.text_lora_spans):
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            reference_latents=ref_latents_list,
            aspect_ratio=args.aspect_ratio,
            video_length=args.video_length,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            enable_sr=False,
            prompt_rewrite=False,
            output_type="pt",
            seed=args.seed,
        )

    if is_main_process:
        print("✅ Video generation complete")

    # Save video (only on main process)
    if is_main_process:
        print(f"\n💾 Saving video to {args.output_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        save_video(output.videos, args.output_path, fps=args.fps)

        print("\n" + "=" * 80)
        print("✨ Done! ✨")
        print("=" * 80)

    # Cleanup distributed
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
