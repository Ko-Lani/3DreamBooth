# OminiControl-style multi-branch attention for HunyuanVideo-1.5
# Adapted from OminiControl (https://github.com/Yuanshi9815/OminiControl)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager
from einops import rearrange

from peft.tuners.tuners_utils import BaseTunerLayer


from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.utils.communications import all_gather, all_to_all_4D
from .modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .modules.modulate_layers import modulate, apply_gate



@dataclass
class ReferenceCondition:
    """
    Reference condition for OminiControl-style generation.

    Args:
        condition: Reference image latent tensor [B, C, T, H, W] or [B, C, H, W]
        position_delta: Spatial offset (dt, dh, dw) for position embedding
        adapter: LoRA adapter name for this condition
    """
    condition: torch.Tensor
    position_delta: Tuple[int, int, int] = (0, 0, 0)  # (t, h, w) offset
    adapter: str = "default"

    def get_position_offset(self) -> Tuple[int, int, int]:
        """Get position offset for 3D rotary embedding."""
        return self.position_delta


@contextmanager
def specify_lora(lora_modules: List[nn.Module], specified_lora: str):
    """
    Context manager to temporarily activate only the specified LoRA adapter.

    Args:
        lora_modules: List of modules that may have LoRA adapters
        specified_lora: Name of the adapter to activate (others will be disabled)
    """
    # Filter valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]

    if not valid_lora_modules:
        yield
        return

    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]

    # Enter context: adjust scaling
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    try:
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales.get(adapter, 0)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    """Clip hidden states to prevent overflow in float16."""
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def get_rotary_pos_embed_with_offset(
    transformer: nn.Module,
    thw: Tuple[int, int, int],
    position_delta: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rotary position embeddings with spatial offset.

    This is crucial for OminiControl-style multi-branch attention where reference
    images need different position encodings than the main image to prevent
    the model from confusing reference content with main content.

    Args:
        transformer: HunyuanVideo transformer with get_rotary_pos_embed method
        thw: Tuple of (T, H, W) dimensions for the sequence
        position_delta: Offset (dt, dh, dw) to add to positions

    Returns:
        (freqs_cos, freqs_sin): Position embeddings with offset applied
    """
    tt, th, tw = thw
    dt, dh, dw = position_delta

    # If no offset, use the standard method
    if dt == 0 and dh == 0 and dw == 0:
        return transformer.get_rotary_pos_embed((tt, th, tw))

    # Get rope parameters from transformer attributes (same as get_rotary_pos_embed)
    target_ndim = 3
    head_dim = transformer.hidden_size // transformer.heads_num
    rope_dim_list = transformer.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

    theta = transformer.rope_theta  # Usually 256.0 for HunyuanVideo

    # Generate position embeddings with offset
    # start: (dt, dh, dw) - offset position
    # stop: (dt + tt, dh + th, dw + tw) - end position
    # num: (tt, th, tw) - number of positions
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        tuple(rope_dim_list),
        (dt, dh, dw),  # start with offset
        (dt + tt, dh + th, dw + tw),  # stop
        (tt, th, tw),  # num
        theta=theta,
        use_real=True,
        theta_rescale_factor=1,  # same as original
    )

    return freqs_cos, freqs_sin


def omini_attention_forward(
    queries: List[torch.Tensor],
    keys: List[torch.Tensor],
    values: List[torch.Tensor],
    group_mask: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
    img_seq_len: int = 0,
) -> List[torch.Tensor]:
    """
    Multi-branch attention with group masking.

    Args:
        queries: List of query tensors [B, L, H, D] for each branch
        keys: List of key tensors [B, L, H, D] for each branch
        values: List of value tensors [B, L, H, D] for each branch
        group_mask: Boolean tensor [num_branches, num_branches] controlling attention
        text_mask: Optional attention mask for text tokens
        img_seq_len: Length of image sequence (for splitting results)

    Returns:
        List of attention outputs for each branch
    """
    num_branches = len(queries)
    attn_outputs = []

    for i, query in enumerate(queries):
        # Collect keys and values from branches that this branch should attend to
        keys_to_attend = []
        values_to_attend = []

        for j, (k, v) in enumerate(zip(keys, values)):
            if group_mask[i, j].item():
                keys_to_attend.append(k)
                values_to_attend.append(v)

        if not keys_to_attend:
            # If no branches to attend to, skip (shouldn't happen normally)
            attn_outputs.append(torch.zeros_like(query).reshape(query.shape[0], query.shape[1], -1))
            continue

        # Concatenate keys and values
        k_cat = torch.cat(keys_to_attend, dim=1)  # [B, total_L, H, D]
        v_cat = torch.cat(values_to_attend, dim=1)  # [B, total_L, H, D]

        # Transpose for scaled_dot_product_attention: [B, H, L, D]
        q_t = query.transpose(1, 2)
        k_t = k_cat.transpose(1, 2)
        v_t = v_cat.transpose(1, 2)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(q_t, k_t, v_t)

        # Transpose back: [B, L, H, D]
        attn_output = attn_output.transpose(1, 2)

        # Reshape to [B, L, H*D]
        b, s, h, d = attn_output.shape
        attn_output = attn_output.reshape(b, s, h * d)

        attn_outputs.append(attn_output)

    return attn_outputs


def omini_double_block_forward(
    block: nn.Module,
    img_hidden_states: List[torch.Tensor],
    txt_hidden_states: torch.Tensor,
    txt_vec: torch.Tensor,
    img_vecs: List[torch.Tensor],
    adapters: List[str],
    group_mask: torch.Tensor,
    freqs_cis_list: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Forward pass for MMDoubleStreamBlock with multi-branch support.

    Args:
        block: MMDoubleStreamBlock instance
        img_hidden_states: List of image hidden states [main_img, ref1, ref2, ...]
        txt_hidden_states: Text hidden states tensor
        txt_vec: Modulation vector for text branch (timestep t)
        img_vecs: Per-branch modulation vectors [main(t), cond1(0), cond2(0), ...]
        adapters: List of LoRA adapter names for each image branch
        group_mask: Attention group mask
        freqs_cis_list: List of rotary position embeddings per branch [(cos, sin), ...]
        text_mask: Text attention mask

    Returns:
        Updated (img_hidden_states_list, txt_hidden_states)
    """
    num_img_branches = len(img_hidden_states)

    # Text modulation - LoRA OFF (text branch is never condition)
    with specify_lora([block.txt_mod.linear], None):
        txt_mod_params = block.txt_mod(txt_vec).chunk(6, dim=-1)
    txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod_params[0], txt_mod_params[1], txt_mod_params[2]
    txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod_params[3], txt_mod_params[4], txt_mod_params[5]

    # Process text branch - LoRA OFF
    txt_modulated = block.txt_norm1(txt_hidden_states)
    txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
    with specify_lora([block.txt_attn_q, block.txt_attn_k, block.txt_attn_v], None):
        txt_q = block.txt_attn_q(txt_modulated)
        txt_k = block.txt_attn_k(txt_modulated)
        txt_v = block.txt_attn_v(txt_modulated)
    txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=block.heads_num)
    txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=block.heads_num)
    txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=block.heads_num)
    txt_q = block.txt_attn_q_norm(txt_q).to(txt_v)
    txt_k = block.txt_attn_k_norm(txt_k).to(txt_v)

    # Process image branches with per-branch LoRA
    img_qs, img_ks, img_vs = [], [], []
    img_mod_params_list = []
    for i, img in enumerate(img_hidden_states):
        # Per-branch modulation (LoRA ON for condition, OFF for main)
        # Each branch uses its own vec: main gets timestep t, conditions get timestep 0
        with specify_lora([block.img_mod.linear], adapters[i]):
            branch_mod = block.img_mod(img_vecs[i]).chunk(6, dim=-1)
        img_mod_params_list.append(branch_mod)

        img_modulated = block.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=branch_mod[0], scale=branch_mod[1])

        with specify_lora([block.img_attn_q, block.img_attn_k, block.img_attn_v], adapters[i]):
            img_q = block.img_attn_q(img_modulated)
            img_k = block.img_attn_k(img_modulated)
            img_v = block.img_attn_v(img_modulated)

        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=block.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=block.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=block.heads_num)
        img_q = block.img_attn_q_norm(img_q).to(img_v)
        img_k = block.img_attn_k_norm(img_k).to(img_v)

        # Apply rotary embedding to image queries and keys (per-branch)
        if freqs_cis_list is not None and i < len(freqs_cis_list):
            freqs_cis = freqs_cis_list[i]
            if freqs_cis is not None:
                img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)

        img_qs.append(img_q)
        img_ks.append(img_k)
        img_vs.append(img_v)

    # Build query, key, value lists for all branches
    # Order: [txt, img0, img1, img2, ...]
    all_queries = [txt_q] + img_qs
    all_keys = [txt_k] + img_ks
    all_values = [txt_v] + img_vs

    # Compute multi-branch attention
    attn_outputs = omini_attention_forward(
        all_queries, all_keys, all_values,
        group_mask=group_mask,
        text_mask=text_mask,
    )

    # Extract attention outputs
    txt_attn = attn_outputs[0]
    img_attns = attn_outputs[1:]

    # Text branch output - LoRA OFF
    with specify_lora([block.txt_attn_proj], None):
        txt_hidden_states = txt_hidden_states + apply_gate(
            block.txt_attn_proj(txt_attn), gate=txt_mod1_gate
        )
    with specify_lora([block.txt_mlp.fc2], None):
        txt_hidden_states = txt_hidden_states + apply_gate(
            block.txt_mlp(modulate(block.txt_norm2(txt_hidden_states), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )
    txt_hidden_states = clip_hidden_states(txt_hidden_states)

    # Image branches output - per-branch LoRA
    img_outputs = []
    for i, (img, img_attn) in enumerate(zip(img_hidden_states, img_attns)):
        branch_mod = img_mod_params_list[i]

        with specify_lora([block.img_attn_proj], adapters[i]):
            img = img + apply_gate(block.img_attn_proj(img_attn), gate=branch_mod[2])

        with specify_lora([block.img_mlp.fc2], adapters[i]):
            img = img + apply_gate(
                block.img_mlp(modulate(block.img_norm2(img), shift=branch_mod[3], scale=branch_mod[4])),
                gate=branch_mod[5],
            )

        img_outputs.append(clip_hidden_states(img))

    return img_outputs, txt_hidden_states


def omini_single_block_forward(
    block: nn.Module,
    hidden_states: List[torch.Tensor],
    branch_vecs: List[torch.Tensor],
    txt_len: int,
    adapters: List[str],
    group_mask: torch.Tensor,
    freqs_cis_list: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    text_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Forward pass for MMSingleStreamBlock with multi-branch support.

    In single blocks, img and txt are concatenated. Each branch in hidden_states
    contains [img_tokens, txt_tokens].

    Args:
        block: MMSingleStreamBlock instance
        hidden_states: List of hidden states (each contains concat of img and txt)
        branch_vecs: Per-branch modulation vectors [main(t), cond1(0), cond2(0), ...]
        txt_len: Length of text sequence
        adapters: List of LoRA adapter names for each branch
        group_mask: Attention group mask
        freqs_cis_list: List of rotary position embeddings per branch [(cos, sin), ...]
        text_mask: Text attention mask

    Returns:
        Updated hidden_states list
    """
    num_branches = len(hidden_states)

    # Process each branch with per-branch modulation
    all_queries, all_keys, all_values = [], [], []
    all_mlp_outputs = []
    branch_gates = []

    for i, x in enumerate(hidden_states):
        # Per-branch modulation: main gets timestep t, conditions get timestep 0
        with specify_lora([block.modulation.linear], adapters[i]):
            mod_shift, mod_scale, mod_gate = block.modulation(branch_vecs[i]).chunk(3, dim=-1)
        branch_gates.append(mod_gate)

        x_mod = modulate(block.pre_norm(x), shift=mod_shift, scale=mod_scale)

        with specify_lora([block.linear1_q, block.linear1_k, block.linear1_v, block.linear1_mlp], adapters[i]):
            q = block.linear1_q(x_mod)
            k = block.linear1_k(x_mod)
            v = block.linear1_v(x_mod)
            mlp = block.linear1_mlp(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=block.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=block.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=block.heads_num)

        # Apply QK-Norm
        q = block.q_norm(q).to(v)
        k = block.k_norm(k).to(v)

        # Split into img and txt parts for rotary embedding
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]

        # Apply rotary embedding only to image part (per-branch)
        if freqs_cis_list is not None and i < len(freqs_cis_list):
            freqs_cis = freqs_cis_list[i]
            if freqs_cis is not None:
                img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)

        # Recombine
        q = torch.cat([img_q, txt_q], dim=1)
        k = torch.cat([img_k, txt_k], dim=1)

        all_queries.append(q)
        all_keys.append(k)
        all_values.append(v)
        all_mlp_outputs.append(mlp)

    # Compute multi-branch attention
    attn_outputs = omini_attention_forward(
        all_queries, all_keys, all_values,
        group_mask=group_mask,
        text_mask=text_mask,
    )

    # Apply output projections and residual connections
    outputs = []
    for i, (x, attn_out, mlp) in enumerate(zip(hidden_states, attn_outputs, all_mlp_outputs)):
        with specify_lora([block.linear2], adapters[i]):
            h = torch.cat([attn_out, block.mlp_act(mlp)], dim=2)
            output = block.linear2(h, None)  # LinearWarpforSingle expects two args

        output = x + apply_gate(output, gate=branch_gates[i])
        outputs.append(clip_hidden_states(output))

    return outputs


def create_group_mask(num_conditions: int, allow_cond_cross_attn: bool = False) -> torch.Tensor:
    """
    Create group mask for multi-branch attention.

    Structure: [txt, main_img, cond1, cond2, ...]

    Args:
        num_conditions: Number of reference conditions
        allow_cond_cross_attn: Whether conditions can attend to each other

    Returns:
        Boolean tensor of shape [num_branches, num_branches]
    """
    num_branches = 2 + num_conditions  # txt + main_img + conditions
    group_mask = torch.ones(num_branches, num_branches, dtype=torch.bool)

    if not allow_cond_cross_attn and num_conditions > 0:
        # Disable cross-attention between condition branches
        # Conditions are at indices 2, 3, 4, ...
        cond_start = 2
        for i in range(cond_start, num_branches):
            for j in range(cond_start, num_branches):
                if i != j:
                    group_mask[i, j] = False

    return group_mask


def omini_transformer_forward(
    transformer: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    text_states: torch.Tensor,
    text_states_2: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    conditions: List[ReferenceCondition] = None,
    adapters: List[str] = None,
    timestep_r: torch.Tensor = None,
    vision_states: torch.Tensor = None,
    guidance: torch.Tensor = None,
    mask_type: str = "t2v",
    extra_kwargs: Optional[Dict[str, Any]] = None,
    freqs_cos: Optional[torch.Tensor] = None,
    freqs_sin: Optional[torch.Tensor] = None,
    return_dict: bool = False,
    disable_sp: bool = False,
) -> Tuple[torch.Tensor, Any]:
    """
    OminiControl-style transformer forward with multi-branch reference conditions.

    Args:
        transformer: HunyuanVideo_1_5_DiffusionTransformer instance
        hidden_states: Main image latents [B, C, T, H, W]
        timestep: Diffusion timestep
        text_states: Text encoder hidden states
        text_states_2: Secondary text encoder hidden states
        encoder_attention_mask: Text attention mask
        conditions: List of ReferenceCondition objects
        adapters: List of LoRA adapter names [main_adapter, cond1_adapter, ...]
        ... (other standard arguments)

    Returns:
        (output, features_list)
    """
    conditions = conditions or []
    num_conditions = len(conditions)

    # Set up adapters
    if adapters is None:
        adapters = [None] + ["default"] * num_conditions

    if guidance is None:
        guidance = torch.tensor(
            [6016.0], device=hidden_states.device, dtype=torch.bfloat16
        )

    img = x = hidden_states
    text_mask = encoder_attention_mask
    txt = text_states

    # Ensure timestep is a tensor
    if isinstance(timestep, (int, float)):
        t = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
    else:
        t = timestep
    bs, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // transformer.patch_size[0],
        oh // transformer.patch_size[1],
        ow // transformer.patch_size[2],
    )
    transformer.attn_param['thw'] = [tt, th, tw]

    # Get rotary position embeddings for main image
    if freqs_cos is None and freqs_sin is None:
        freqs_cos, freqs_sin = transformer.get_rotary_pos_embed((tt, th, tw))

    # Generate position embeddings for reference conditions with spatial offset
    # Reference images get W offset so they don't share positions with main content
    # This is crucial for multi-branch attention to distinguish reference from main
    ref_freqs_list = []
    if conditions:  # Only process if there are conditions
        ref_position_offset = tw + 8  # gap of 8 positions in width dimension
        for i, cond in enumerate(conditions):
            cond_tensor = cond.condition
            if cond_tensor.ndim == 4:
                cond_tensor = cond_tensor.unsqueeze(2)

            # Get reference dimensions
            # Ensure at least 1 for each dimension to avoid empty tensors
            ref_t = max(1, cond_tensor.shape[2] // transformer.patch_size[0])
            ref_h = max(1, cond_tensor.shape[3] // transformer.patch_size[1])
            ref_w = max(1, cond_tensor.shape[4] // transformer.patch_size[2])

            # Position delta: use spatial offset in W dimension
            # Each reference gets offset by (i+1) * ref_position_offset
            w_offset = (i + 1) * ref_position_offset
            position_delta = cond.position_delta if cond.position_delta != (0, 0, 0) else (0, 0, w_offset)

            try:
                ref_freqs = get_rotary_pos_embed_with_offset(
                    transformer,
                    thw=(ref_t, ref_h, ref_w),
                    position_delta=position_delta,
                )
                ref_freqs_list.append(ref_freqs)
            except Exception as e:
                # Fallback: use same position embedding as main (not ideal but works)
                print(f"Warning: Could not generate offset position embedding for reference {i}: {e}")
                print(f"Using main position embedding for reference {i}")
                ref_freqs_list.append((freqs_cos, freqs_sin))

    # Patch embed main image (LoRA OFF for main branch)
    with specify_lora([transformer.img_in.proj], adapters[0]):
        img = transformer.img_in(img)

    # Patch embed conditions (LoRA ON for condition branches)
    cond_hidden_states = []
    for i, cond in enumerate(conditions):
        cond_tensor = cond.condition
        if cond_tensor.ndim == 4:
            cond_tensor = cond_tensor.unsqueeze(2)  # Add temporal dim: [B, C, 1, H, W]

        # Add empty condition channels to match main input format
        # Main input is [latent(32) + cond(33)] = 65 channels
        # Reference latent is 32 channels, need to add 33 empty channels
        b_c, c_c, t_c, h_c, w_c = cond_tensor.shape
        empty_cond = torch.zeros(
            [b_c, c_c + 1, t_c, h_c, w_c],
            device=cond_tensor.device,
            dtype=cond_tensor.dtype
        )
        cond_with_empty = torch.cat([cond_tensor, empty_cond], dim=1)

        with specify_lora([transformer.img_in.proj], adapters[i + 1]):
            cond_hidden = transformer.img_in(cond_with_empty)
        cond_hidden_states.append(cond_hidden)

    # Handle sequence parallelism for main image only
    # Reference conditions must NOT be chunked: each GPU needs the full reference
    # to provide consistent conditioning across all video frame chunks.
    parallel_dims = get_parallel_state()
    sp_enabled = parallel_dims.sp_enabled and not disable_sp
    if sp_enabled:
        sp_size = parallel_dims.sp
        sp_rank = parallel_dims.sp_rank
        if img.shape[1] % sp_size != 0:
            n_token = img.shape[1]
            assert n_token > (n_token // sp_size + 1) * (sp_size - 1)
        img = torch.chunk(img, sp_size, dim=1)[sp_rank]
        freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
        freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

        # Do NOT chunk cond_hidden_states or ref_freqs_list:
        # Each GPU must see the full reference image so that all video
        # frame chunks receive identical reference conditioning.

    # Prepare modulation vectors
    vec = transformer.time_in(t)
    # Condition branches use timestep=0 (clean, non-noisy reference image)
    # In OminiControl, conditions are clean images so their timestep embedding should be 0
    t_cond = torch.zeros_like(t)
    vec_cond = transformer.time_in(t_cond)
    if text_states_2 is not None:
        vec_2 = transformer.vector_in(text_states_2)
        vec = vec + vec_2
        vec_cond = vec_cond + vec_2
    if transformer.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        g_emb = transformer.guidance_in(guidance)
        vec = vec + g_emb
        vec_cond = vec_cond + g_emb
    if timestep_r is not None:
        t_r_emb = transformer.time_r_in(timestep_r)
        vec = vec + t_r_emb
        vec_cond = vec_cond + t_r_emb

    # Per-branch modulation vectors:
    # img_vecs[0] = main image (timestep t), img_vecs[1:] = conditions (timestep 0)
    img_vecs = [vec] + [vec_cond] * num_conditions

    # Embed text tokens
    if transformer.text_projection == "linear":
        txt = transformer.txt_in(txt)
    elif transformer.text_projection == "single_refiner":
        txt = transformer.txt_in(txt, t, text_mask if transformer.use_attention_mask else None)

    if transformer.cond_type_embedding is not None:
        cond_emb = transformer.cond_type_embedding(
            torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
        )
        txt = txt + cond_emb

    # Handle byT5
    if transformer.glyph_byT5_v2 and extra_kwargs is not None:
        byt5_text_states = extra_kwargs.get("byt5_text_states")
        byt5_text_mask = extra_kwargs.get("byt5_text_mask")
        if byt5_text_states is not None:
            byt5_txt = transformer.byt5_in(byt5_text_states)
            if transformer.cond_type_embedding is not None:
                cond_emb = transformer.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = transformer.reorder_txt_token(
                byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
            )

    # Handle vision states (for i2v)
    if transformer.vision_in is not None and vision_states is not None:
        extra_encoder_hidden_states = transformer.vision_in(vision_states)
        if mask_type == "t2v" and torch.all(vision_states == 0):
            extra_attention_mask = torch.zeros(
                (bs, extra_encoder_hidden_states.shape[1]),
                dtype=text_mask.dtype,
                device=text_mask.device,
            )
            extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
        else:
            extra_attention_mask = torch.ones(
                (bs, extra_encoder_hidden_states.shape[1]),
                dtype=text_mask.dtype,
                device=text_mask.device,
            )
        if transformer.cond_type_embedding is not None:
            cond_emb = transformer.cond_type_embedding(
                2 * torch.ones_like(
                    extra_encoder_hidden_states[:, :, 0],
                    dtype=torch.long,
                    device=extra_encoder_hidden_states.device,
                )
            )
            extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
        txt, text_mask = transformer.reorder_txt_token(
            extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
        )

    # Build freqs_cis_list: [main_freqs, ref1_freqs, ref2_freqs, ...]
    main_freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
    freqs_cis_list = [main_freqs_cis] + ref_freqs_list

    # Create group mask
    # Structure: [txt, main_img, cond1, cond2, ...]
    group_mask = create_group_mask(num_conditions, allow_cond_cross_attn=False)
    group_mask = group_mask.to(hidden_states.device)

    # Combine image hidden states: [main_img, cond1, cond2, ...]
    img_hidden_states = [img] + cond_hidden_states

    # Build adapters list: [main_adapter, cond1_adapter, ...]
    all_adapters = adapters

    # Pass through double-stream blocks
    for index, block in enumerate(transformer.double_blocks):
        img_hidden_states, txt = omini_double_block_forward(
            block,
            img_hidden_states=img_hidden_states,
            txt_hidden_states=txt,
            txt_vec=vec,
            img_vecs=img_vecs,
            adapters=all_adapters,
            group_mask=group_mask,
            freqs_cis_list=freqs_cis_list,
            text_mask=text_mask,
        )

    txt_seq_len = txt.shape[1]
    img_seq_len = img_hidden_states[0].shape[1]

    # Merge image and text for single-stream blocks
    # Each branch: concat(img_branch, txt)
    merged_hidden_states = []
    for img_h in img_hidden_states:
        merged_hidden_states.append(torch.cat([img_h, txt], dim=1))

    # Pass through single-stream blocks
    if len(transformer.single_blocks) > 0:
        for index, block in enumerate(transformer.single_blocks):
            merged_hidden_states = omini_single_block_forward(
                block,
                hidden_states=merged_hidden_states,
                branch_vecs=img_vecs,
                txt_len=txt_seq_len,
                adapters=all_adapters,
                group_mask=group_mask,
                freqs_cis_list=freqs_cis_list,
                text_mask=text_mask,
            )

    # Extract main image output (first branch)
    x = merged_hidden_states[0]
    img = x[:, :img_seq_len, ...]

    # Final Layer
    img = transformer.final_layer(img, vec)
    if sp_enabled:
        img = all_gather(img, dim=1, group=parallel_dims.sp_group)
    img = transformer.unpatchify(img, tt, th, tw)

    return (img, None)


def encode_reference_images(
    vae,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Encode reference images using VAE.

    Args:
        vae: VAE model
        images: Images tensor [B, C, H, W] in range [-1, 1]
        device: Target device
        dtype: Target dtype

    Returns:
        Encoded latents
    """
    if images.ndim == 4:
        images = images.unsqueeze(2)  # Add temporal dim: [B, C, 1, H, W]

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        with vae.memory_efficient_context() if hasattr(vae, 'memory_efficient_context') else torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            else:
                latents = latents * vae.config.scaling_factor

    return latents
