from typing import Callable, Optional, Tuple, Union

import torch


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor,
    b_transposed: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_src, _ = a.shape
    original_dtype = a.dtype

    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(batch_size, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(batch_size, num_src, device=a.device, dtype=torch.long)

    for start in range(0, num_src, chunk_size):
        end = min(start + chunk_size, num_src)
        scores_chunk = torch.bmm(a_bf16[:, start:end, :], b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        node_max[:, start:end] = chunk_max_bf16.to(original_dtype)
        node_idx[:, start:end] = chunk_idx

    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    mode: str = "mean",
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    if extra_tensors is not None:
        return x, extra_tensors
    return x


def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    num_special_tokens: int = 5,
    merge_eligible_mask: Optional[torch.Tensor] = None,
) -> Tuple[Callable, Callable]:
    """
    FastVGGT-style bipartite token merging over a flattened multi-frame sequence.

    Each frame is expected to contain `num_special_tokens` prefix tokens followed
    by an h-by-w patch grid. The first frame and every frame's prefix tokens are
    kept as destination tokens. Register-attention blocks should not call this
    function; they operate on special tokens only and must stay unmerged.
    """
    batch_size, num_tokens_total, _ = metric.shape
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather
    tokens_per_img = w * h + num_special_tokens
    num_imgs = num_tokens_total // tokens_per_img
    if tokens_per_img * num_imgs != num_tokens_total:
        raise ValueError(
            "Token count does not match "
            f"(w*h+num_special_tokens)*num_imgs: "
            f"{num_tokens_total} vs {tokens_per_img}*{num_imgs}"
        )

    with torch.no_grad():
        if merge_eligible_mask is not None:
            if merge_eligible_mask.shape != (batch_size, num_tokens_total):
                raise ValueError(
                    "merge_eligible_mask must have shape "
                    f"{(batch_size, num_tokens_total)}, got {tuple(merge_eligible_mask.shape)}"
                )
            if batch_size != 1:
                raise ValueError("Selective merge masks currently require batch size 1")
            merge_eligible_mask = merge_eligible_mask.to(device=metric.device, dtype=torch.bool)
        if enable_protection:
            num_protected = max(1, int(num_tokens_total * 0.1))
            step = max(1, num_tokens_total // num_protected)
            protected_indices = torch.arange(0, num_tokens_total, step, device=metric.device)[
                :num_protected
            ]
        else:
            protected_indices = None
            num_protected = 0

        idx_buffer_seq = torch.zeros(num_tokens_total, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx

        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        if num_imgs > 1:
            special_indices = torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            special_indices = special_indices[:, None] + torch.arange(
                num_special_tokens,
                device=metric.device,
            )
            idx_buffer_seq[special_indices.flatten()] = -1

            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(effective_grid_size, device=metric.device, dtype=torch.int64)
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
                    + num_special_tokens
                )
                grid_indices = grid_starts[:, None] + torch.arange(effective_grid_size, device=metric.device)
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(num_imgs - 1)
            else:
                total_other_imgs = num_imgs - 1
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(total_other_imgs, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )
                scatter_src = -torch.ones(total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64)
                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )
                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                for i in range(total_other_imgs):
                    img_idx = i + 1
                    grid_start = img_idx * tokens_per_img + num_special_tokens
                    flat_view = idx_buffer_batch[i, :effective_h, :effective_w].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = flat_view

        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        a_idx = rand_idx[:, num_dst_orig:, :]
        b_idx = rand_idx[:, :num_dst_orig, :]

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        def split(x):
            channels = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(batch_size, num_src, channels))
            dst = gather(x, dim=1, index=b_idx.expand(batch_size, num_dst, channels))
            if enable_protection:
                protected = gather(
                    x,
                    dim=1,
                    index=protected_idx.expand(batch_size, num_protected_actual, channels),
                )
                return src, dst, protected
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, _ = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            valid_mask = ~protected_mask_src[edge_flat]
            if merge_eligible_mask is not None:
                eligible_src = merge_eligible_mask[0, src_indices]
                valid_mask &= eligible_src[edge_flat]
            valid_edges = edge_flat[valid_mask]

            r_actual = min(r, valid_edges.shape[0])
            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            if merge_eligible_mask is not None:
                src_indices = a_idx[0, :, 0]
                edge_flat = edge_idx[0, :, 0]
                eligible_src = merge_eligible_mask[0, src_indices]
                valid_edges = edge_flat[eligible_src[edge_flat]]
                invalid_edges = edge_flat[~eligible_src[edge_flat]]
                r_actual = min(r, valid_edges.shape[0])
                src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
                unm_idx = torch.cat((valid_edges[r_actual:], invalid_edges)).unsqueeze(0).unsqueeze(-1)
                r = r_actual
            else:
                unm_idx = edge_idx[..., r:, :]
                src_idx = edge_idx[..., :r, :]
                r_actual = r

        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        selected_source_indices = gather(
            a_idx.expand(batch_size, a_idx.shape[1], 1),
            dim=1,
            index=src_idx,
        )
        selected_destination_indices = gather(
            b_idx.expand(batch_size, b_idx.shape[1], 1),
            dim=1,
            index=dst_idx,
        )
        r = r_actual

    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, _, channels = src.shape
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, channels))
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, channels))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, channels), src, reduce=mode)

        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            extra_channels = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)

            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, extra_channels))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, extra_channels))
            dst_e = dst_e.scatter_reduce(-2, dst_idx.expand(n, src_len, extra_channels), src_e_r, reduce=mode)
            merged_extra_1 = (
                torch.cat([unm_e, dst_e, protected_e], dim=1)
                if enable_protection
                else torch.cat([unm_e, dst_e], dim=1)
            )

        if extra_tensors_2 is not None:
            extra_channels_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)

            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, extra_channels_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, extra_channels_2))
            dst_e2 = dst_e2.scatter_reduce(
                -2,
                dst_idx.expand(n, src_len, extra_channels_2),
                src_e2_r,
                reduce=mode,
            )
            merged_extra_2 = (
                torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
                if enable_protection
                else torch.cat([unm_e2, dst_e2], dim=1)
            )

        main_result = (
            torch.cat([unm, dst, protected], dim=1)
            if enable_protection
            else torch.cat([unm, dst], dim=1)
        )

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        if merged_extra_1 is not None:
            return main_result, merged_extra_1
        return main_result

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]

        if enable_protection:
            protected = x[..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :]

        _, _, channels = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(batch_size, src_len, channels))
        out = torch.zeros(batch_size, num_tokens_total, channels, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(batch_size, num_dst, channels), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(batch_size, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(
                batch_size,
                unm_len,
                channels,
            ),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(a_idx.expand(batch_size, a_idx.shape[1], 1), dim=1, index=src_idx).expand(
                batch_size,
                src_len,
                channels,
            ),
            src=src,
        )

        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(batch_size, num_protected_actual, channels),
                src=protected,
            )

        return out

    merge.selected_source_indices = selected_source_indices
    merge.selected_destination_indices = selected_destination_indices
    return merge, unmerge
