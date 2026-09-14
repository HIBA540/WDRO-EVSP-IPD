"""Matrice de distance et voisinage."""
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

from config.settings import cfg


def build_distance_matrix_exact(Xi_norm):
    Xi_t = np.nan_to_num(Xi_norm, nan=0., posinf=5., neginf=-5.)
    S = Xi_t.shape[0]
    D_full = cdist(Xi_t, Xi_t, 'euclidean')
    np.fill_diagonal(D_full, 0.)
    print(f"    [D_sp] {S}x{S}  d_min={D_full[D_full>0].min():.4f}  d_mean={D_full[D_full>0].mean():.4f}")
    return D_full


def build_neighbor_map(D_sp: np.ndarray, k_neighbors: int):
    S = D_sp.shape[0]
    neighbor_map = {}
    for s in range(S):
        order = np.argsort(D_sp[s])
        order = order[order != s]
        neighbor_map[s] = [s] + order[:k_neighbors].tolist()
    return neighbor_map


def _select_recourse_scenarios(anchor_indices, neighbor_map, max_scenarios,
                                seed=0, verbose=True):
    coverage = Counter()
    for s in anchor_indices:
        for j in neighbor_map.get(s, []):
            coverage[j] += 1

    all_needed = sorted(coverage.keys())
    n_before = len(all_needed)
    anchors_set = set(anchor_indices)
    anchors_needed = sorted(anchors_set & set(all_needed))
    non_anchor_needed = sorted(set(all_needed) - anchors_set)

    diagnostics = dict(
        n_anchors=len(anchor_indices),
        n_union_before_truncation=n_before,
        max_scenarios=max_scenarios,
        n_anchors_covered=0,
        n_neighbors_kept=0,
        fraction_kept=1.0,
    )

    if n_before <= max_scenarios:
        needed = all_needed
        diagnostics['n_anchors_covered'] = len(anchors_needed)
        diagnostics['n_neighbors_kept'] = len(non_anchor_needed)
        diagnostics['fraction_kept'] = 1.0
        if verbose:
            print(f"    union={n_before} <= max_scenarios={max_scenarios}")
        return needed, diagnostics

    if len(anchors_needed) <= max_scenarios:
        selected = list(anchors_needed)
        budget_left = max_scenarios - len(selected)
    else:
        anchors_ranked = sorted(anchors_needed, key=lambda j: (-coverage[j], j))
        selected = anchors_ranked[:max_scenarios]
        budget_left = 0
        if verbose:
            print(f"    WARNING: {len(anchors_needed)} ancres > {max_scenarios}")

    if budget_left > 0 and non_anchor_needed:
        ranked = sorted(non_anchor_needed, key=lambda j: (-coverage[j], j))
        selected += ranked[:budget_left]

    needed = sorted(selected)
    diagnostics['n_anchors_covered'] = len(set(needed) & anchors_set)
    diagnostics['n_neighbors_kept'] = len(needed) - diagnostics['n_anchors_covered']
    diagnostics['fraction_kept'] = len(needed) / max(n_before, 1)

    if verbose:
        print(f"    union={n_before} -> tronque a {len(needed)} "
              f"(fraction={diagnostics['fraction_kept']:.3f})")
    return needed, diagnostics