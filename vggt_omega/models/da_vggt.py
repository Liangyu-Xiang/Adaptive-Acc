"""Shared DA-VGGT partition primitives for VGGT-Omega."""
from __future__ import annotations
import numpy as np

def cosine_similarity(features):
    x = features.float().numpy(); x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.clip(x @ x.T, -1.0, 1.0)

def diversity_partition(sim, chunk_size, anchors, iters=5):
    n = len(sim); groups = [[] for _ in range((n + chunk_size - 1) // chunk_size)]
    cap = max(1, (n + len(groups) - 1) // len(groups)); utility = 1.0 - sim; np.fill_diagonal(utility, 0)
    remaining = set(range(n)) - set(anchors)
    for group in groups:
        while remaining and len(group) < cap:
            _, item = max(((utility[i, group].sum() if group else utility[i].sum(), i) for i in remaining))
            group.append(item); remaining.remove(item)
    for item in remaining: min(groups, key=len).append(item)
    for _ in range(iters):
        changed = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                best = (0., None, None)
                for ia, x in enumerate(groups[a]):
                    for ib, y in enumerate(groups[b]):
                        gain = utility[y, groups[a]].sum() + utility[x, groups[b]].sum() - utility[x, groups[a]].sum() - utility[y, groups[b]].sum()
                        if gain > best[0]: best = (gain, ia, ib)
                if best[1] is not None:
                    ia, ib = best[1:]; groups[a][ia], groups[b][ib] = groups[b][ib], groups[a][ia]; changed = True
        if not changed: break
    aset = set(anchors)
    return [[anchors[0], *[x for x in group if x not in aset]] for group in groups]

def pseudo_positions(sim, indices, positions, gamma):
    n = len(sim); result = np.zeros((n, 3)); known = np.asarray(indices); result[known] = positions; known_set = set(indices)
    for i in range(n):
        if i in known_set: continue
        logits = sim[i, known] / max(gamma, 1e-6); logits -= logits.max(); weight = np.exp(logits); weight /= weight.sum(); result[i] = weight @ positions
    return result

def pose_weighted_similarity(sim, pseudo, tau=None):
    distance = np.linalg.norm(pseudo[:, None] - pseudo[None, :], axis=-1)
    tau = float(np.median(distance[distance > 0])) if tau is None and np.any(distance > 0) else (tau or 1.)
    return sim * np.exp(-distance / max(tau, 1e-6))
