# meta_hierarchical.py
# Lightweight Meta controller wrapper around DOAR (DoubleOracle) low-level policy.
# Updated to include Q-value caching for node-partitioned critic evaluations with
# TTL, periodic flushing, k-hop invalidation, and random re-evaluation.

from __future__ import annotations

import math
import copy
import collections
import random
import tempfile
import os
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from strategy import Strategy  # type: ignore


def _feature_adjacency(self, env) -> torch.Tensor:
    """
    Adjacency used for structural features.

    - Defender: full graph (they know the network topology).
    - Attacker: adjacency masked by the visibility mask, so degrees
        only reflect the discovered / visible subgraph.
    """
    A = adjacency_matrix_from_env(env)

    # Defender can keep full topology
    if self.role != "attacker":
        return A

    # Attacker: apply visibility mask to rows/cols
    v = build_visibility_mask(env, self.role)  # (M,) float tensor on CPU
    v = (v > 0.5).to(A.device)  # boolean mask on same device as A

    if getattr(A, "is_sparse", False):
        co = A.coalesce()
        idx = co.indices()   # [2, nnz]
        vals = co.values()   # [nnz]
        rows, cols = idx[0], idx[1]

        keep = v[rows] & v[cols]
        if keep.sum() == 0:
            # no visible edges; keep self-loops on visible nodes so degree>0
            M = A.size(0)
            vis_idx = torch.nonzero(v, as_tuple=False).view(-1)
            new_rows = vis_idx
            new_cols = vis_idx
            new_vals = torch.ones_like(vis_idx, dtype=torch.float32)
            new_idx = torch.stack([new_rows, new_cols], dim=0)
            return torch.sparse_coo_tensor(new_idx, new_vals, (M, M)).coalesce()

        new_idx = idx[:, keep]
        new_vals = vals[keep]
        return torch.sparse_coo_tensor(new_idx, new_vals, A.shape).coalesce()
    else:
        # dense: zero out invisible rows/cols
        v_float = v.float()
        v_col = v_float.view(-1, 1)
        v_row = v_float.view(1, -1)
        return A * v_col * v_row


# -------------------------
# adjacency / visibility helpers (minimal, compatible)
# -------------------------
def adjacency_matrix_from_env(env) -> torch.Tensor:
    M = int(getattr(env, "Max_network_size", 0))
    g = getattr(env.simulator, "subnet", None) or getattr(env.simulator, "graph", None) or getattr(env, "graph", None)
    rows, cols, vals = [], [], []
    try:
        if getattr(env.simulator, "subnet", None) is not None:
            g = env.simulator.subnet.graph if hasattr(env.simulator.subnet, "graph") else env.simulator.subnet
    except Exception:
        pass
    g_obj = getattr(env, "graph", None) or g
    if g_obj is not None:
        try:
            eds = getattr(g_obj, "get_edgelist", lambda: [])()
            for (u, v) in eds:
                rows.append(int(u)); cols.append(int(v)); vals.append(1.0)
                rows.append(int(v)); cols.append(int(u)); vals.append(1.0)
        except Exception:
            try:
                for u, v in g_obj.edges():
                    rows.append(int(u)); cols.append(int(v)); vals.append(1.0)
                    rows.append(int(v)); cols.append(int(u)); vals.append(1.0)
            except Exception:
                pass
    for i in range(M):
        rows.append(i); cols.append(i); vals.append(1.0)

    if M <= 1000:
        A = torch.zeros((M, M), dtype=torch.float32)
        if rows:
            idx = np.vstack([rows, cols]).astype(int)
            A_np = np.zeros((M, M), dtype=np.float32)
            A_np[idx[0], idx[1]] = 1.0
            A = torch.tensor(A_np, dtype=torch.float32)
        else:
            A.fill_diagonal_(1.0)
        return A
    else:
        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, (M, M)).coalesce()
        else:
            idx = torch.arange(0, M, dtype=torch.long)
            indices = torch.stack([idx, idx], dim=0)
            values = torch.ones(M, dtype=torch.float32)
            return torch.sparse_coo_tensor(indices, values, (M, M)).coalesce()


def build_visibility_mask(env, role: str) -> torch.Tensor:
    M = int(getattr(env, "Max_network_size", 0))
    v = torch.zeros(M, dtype=torch.float32)
    devices = getattr(env, "_get_ordered_devices", lambda: [])()
    if not devices:
        devices = getattr(env, "devices", []) or getattr(env, "_devices", [])
    for i, d in enumerate(devices):
        if role == "attacker":
            visible = (getattr(d, "Known_to_attacker", False)
                       and getattr(d, "attacker_owned", False)
                       and not getattr(d, "Not_yet_added", False))
        else:
            visible = (not getattr(d, "Not_yet_added", False)
                       and not getattr(d, "attacker_owned", False))
        v[i] = 1.0 if visible else 0.0
    return v

# -------------------------
# small structural featurizer
# -------------------------
class StructuralNodeFeaturizer(nn.Module):
    def __init__(self, M: int, id_dim: int = 16, use_degree: bool = True):
        super().__init__()
        self.M = int(M)
        self.id_emb = nn.Embedding(self.M, id_dim)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.use_degree = use_degree

    def forward_indices(self, env, indices: List[int], A: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(indices) == 0:
            dim = self.id_emb.embedding_dim + (1 if self.use_degree else 0) + 2
            return torch.zeros((0, dim), dtype=torch.float32)
        idx_t = torch.tensor(indices, dtype=torch.long)
        emb = self.id_emb(idx_t)
        if A is None:
            A = adjacency_matrix_from_env(env)
        if getattr(A, "is_sparse", False):
            deg = torch.sparse.sum(A, dim=1).to_dense().cpu().numpy()
        else:
            deg = A.sum(dim=1).cpu().numpy()
        deg_sel = torch.tensor(deg[indices], dtype=torch.float32).unsqueeze(1)
        if deg_sel.max() > 0:
            deg_sel = deg_sel / deg_sel.max()
        visible_flags = []
        owned_flags = []
        devices = getattr(env, "_get_ordered_devices", lambda: [])()
        if not devices:
            devices = getattr(env, "devices", []) or getattr(env, "_devices", [])
        for i in indices:
            try:
                d = devices[i]
                vis = 1.0 if getattr(d, "Known_to_attacker", False) else 0.0
                owned = 1.0 if getattr(d, "attacker_owned", False) else 0.0
            except Exception:
                vis = 0.0; owned = 0.0
            visible_flags.append(vis); owned_flags.append(owned)
        vis_t = torch.tensor(visible_flags, dtype=torch.float32).unsqueeze(1)
        own_t = torch.tensor(owned_flags, dtype=torch.float32).unsqueeze(1)

        parts = [emb]
        if self.use_degree:
            parts.append(deg_sel)
        parts.append(vis_t); parts.append(own_t)
        return torch.cat(parts, dim=1)

# -------------------------
# small projector / controller network
# -------------------------
class StateProjector(nn.Module):
    def __init__(self, state_dim: int, proj_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, proj_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        p = self.fc2(x)
        return p

# -------------------------
# small LRU cache helper class
# -------------------------
class _LRUCache:
    def __init__(self, maxsize: int = 50000):
        self.maxsize = int(maxsize)
        self._d = collections.OrderedDict()

    def get(self, key, default=None):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return default

    def set(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
            self._d[key] = value
            return
        self._d[key] = value
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def pop_prefix_node(self, node_idx: int):
        # remove all keys where the second element equals node_idx:
        # keys are tuples: (state_key, node_idx, atype, exploit_idx, app_idx)
        to_remove = [k for k in list(self._d.keys())
                     if isinstance(k, tuple) and len(k) > 1 and k[1] == node_idx]
        for k in to_remove:
            self._d.pop(k, None)

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)

# -------------------------
# DOAR low-level adapter
# -------------------------
class DOARLowLevelPolicy:
    def __init__(self, oracle, role: str):
        self.oracle = oracle
        self.role = role
        self.device = oracle.device
        self.D = oracle.D_init
        self.E = oracle.E_init
        self.A = oracle.A_init

    def select_action(self, state: np.ndarray, env=None):
        if self.role == "defender":
            ddpg = self.oracle.defender_ddpg
            n_types = self.oracle.n_def_types
        else:
            ddpg = self.oracle.attacker_ddpg
            n_types = self.oracle.n_att_types

        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw = ddpg['actor'](st).cpu().numpy()[0]

        action = self.oracle.decode_action(
            raw,
            num_action_types=n_types,
            num_device_indices=self.D,
            num_exploit_indices=self.E,
            num_app_indices=self.A,
            state_tensor=st,
            actor=ddpg['actor'],
            critic=ddpg['critic']
        )
        at, exps, devs, ai = action
        return [(int(at),
                 np.array(exps, dtype=int),
                 np.array(devs, dtype=int),
                 int(ai))]

# -------------------------
# MetaHierarchicalBestResponse
# -------------------------
class MetaHierarchicalBestResponse:
    def __init__(self,
                 oracle,
                 role: str,
                 embed_dim: int = 32,
                 id_dim: int = 16,
                 proj_dim: int = 32,
                 select_k: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 replay_capacity: int = 2000,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 max_cache_size: int = 50000,
                 cache_state_round_decimals: int = 3,
                 cache_ttl: Optional[int] = 50,
                 cache_flush_interval: Optional[int] = 200,
                 cache_resample_prob: float = 0.01):
        self.oracle = oracle
        self.role = role
        self.env = copy.deepcopy(oracle.env)
        self.device = device or oracle.device
        self.seed = getattr(oracle, "seed", 0)
        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)

        self.state_dim = (self.env._get_defender_state().shape[0]
                          if role == "defender" else self.env._get_attacker_state().shape[0])
        self.M = int(self.env.Max_network_size)
        self.alpha = self.env.alpha
        self.k_hop_invalidate = self.env.khop
        

        self.struct_feats = StructuralNodeFeaturizer(self.M, id_dim=id_dim, use_degree=True).to(self.device)
        struct_out_dim = self.struct_feats.id_emb.embedding_dim + (1 if self.struct_feats.use_degree else 0) + 2
        self.node_proj = nn.Sequential(
            nn.Linear(struct_out_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        ).to(self.device)

        self.proj_dim = proj_dim
        self.state_proj = StateProjector(self.state_dim, proj_dim=self.proj_dim).to(self.device)
        self.target_state_proj = copy.deepcopy(self.state_proj).to(self.device)
        self.target_state_proj.eval()

        self.node_bias = nn.Parameter(torch.zeros(1, device=self.device))

        self.opt = optim.Adam(
            list(self.state_proj.parameters()) + list(self.node_proj.parameters()) + [self.node_bias],
            lr=lr
        )

        # node embeddings cache (structural) and dirty flags
        self.E_cache = torch.zeros((self.M, self.proj_dim), dtype=torch.float32, device=self.device)
        self.node_dirty = np.ones((self.M,), dtype=bool)
        print("Alpha is: "+str(self.alpha))
        if select_k is None:
            self.select_k = max(1, int(math.ceil(self.alpha * math.log10(max(10, self.M)))))
        else:
            self.select_k = int(select_k)
        print("K is currently: "+str(self.select_k))

        # low-level policy adapter (calls DO actor+critic decode)
        self.low_policy = DOARLowLevelPolicy(self.oracle, self.role)

        # replay + controller training
        self.replay = collections.deque(maxlen=replay_capacity)
        self.batch_size = int(batch_size)
        self.gamma = 0.99
        self._train_steps = 0

        # Q-value cache: keys are (state_key, node_idx, atype, exploit_idx, app_idx) -> (q, birth_step)
        self._q_cache = _LRUCache(maxsize=int(max_cache_size))
        self._last_state_key = None
        self._cache_state_round_decimals = int(cache_state_round_decimals)

        # cache staleness controls
        self._cache_step = 0
        self._cache_ttl = cache_ttl if (cache_ttl is None or cache_ttl > 0) else None
        self._cache_flush_interval = cache_flush_interval if (cache_flush_interval is None or cache_flush_interval > 0) else None
        self._cache_resample_prob = float(max(0.0, min(1.0, cache_resample_prob)))

    # -------------------------
    # persistence
    # -------------------------
    def save(self, path: str):
        d = {
            "state_proj": self.state_proj.state_dict(),
            "node_proj": self.node_proj.state_dict(),
            "struct_feats": self.struct_feats.state_dict(),
            "node_bias": self.node_bias.detach().cpu(),
            "select_k": self.select_k,
            "M": self.M,
        }
        torch.save(d, path)

    def load(self, path: str, strict: bool = False):
        data = torch.load(path, map_location=self.device)
        try:
            self.state_proj.load_state_dict(data["state_proj"], strict=not strict)
            self.node_proj.load_state_dict(data["node_proj"], strict=not strict)
            self.struct_feats.load_state_dict(data["struct_feats"], strict=not strict)
            nb = data.get("node_bias", None)
            if nb is not None:
                self.node_bias.data.copy_(nb.to(self.device))
            self.select_k = int(data.get("select_k", self.select_k))
            self.M = int(data.get("M", self.M))
        except Exception:
            pass

    # -------------------------
    # embeddings
    # -------------------------
    def _recompute_embeddings_indices(self, env, indices: List[int]):
        if not indices:
            return
        with torch.no_grad():
            A_feat = self._feature_adjacency(env)  # <-- visibility-masked for attacker
            X = self.struct_feats.forward_indices(env, indices, A=A_feat)
            X = X.to(self.device)
            E = self.node_proj(X)  # (len(indices), proj_dim)
            if E.size(1) != self.proj_dim:
                W = torch.randn((E.size(1), self.proj_dim), device=self.device) * 0.02
                E = E @ W
            self.E_cache[indices] = E


    def _ensure_all_embeddings(self, env):
        all_idx = list(range(self.M))
        self._recompute_embeddings_indices(env, all_idx)
        self.node_dirty[:] = False

    # -------------------------
    # selection
    # -------------------------
    def select_devices(self, env, state_vec: np.ndarray, visible_mask: Optional[np.ndarray] = None, k: Optional[int] = None) -> List[int]:
        if k is None:
            k = self.select_k

        dirty_idx = np.nonzero(self.node_dirty)[0].tolist()
        if dirty_idx:
            self._recompute_embeddings_indices(env, dirty_idx)
            self.node_dirty[dirty_idx] = False

        s_t = torch.tensor(np.asarray(state_vec, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            proj = self.state_proj(s_t).squeeze(0)
        scores = (self.E_cache @ proj).cpu().numpy().astype(float) + float(self.node_bias.detach().cpu().item())

        if visible_mask is None:
            try:
                v_t = build_visibility_mask(env, self.role).cpu().numpy()
                visible_mask = (v_t > 0.5).astype(bool)
            except Exception:
                visible_mask = np.ones((self.M,), dtype=bool)
        cand_idx = np.nonzero(visible_mask)[0]
        if cand_idx.size == 0:
            return []

        cand_scores = scores[cand_idx]
        k2 = min(k, len(cand_idx))
        if k2 <= 0:
            return []
        order = np.argpartition(-cand_scores, k2-1)[:k2]
        top_indices = cand_idx[order]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [int(x) for x in top_indices.tolist()]

    # -------------------------
    # internal: quantize projection -> state_key
    # -------------------------
    def _proj_state_key(self, proj_tensor: torch.Tensor, decimals: Optional[int] = None) -> int:
        if decimals is None:
            decimals = self._cache_state_round_decimals
        arr = proj_tensor.detach().cpu().numpy()
        arr_q = np.round(arr, decimals=decimals)
        return hash(arr_q.tobytes())

    # -------------------------
    # internal: invalidate node from cache
    # -------------------------
    def _invalidate_node_cache(self, node_idx: int):
        """
        Remove all cached Q-entries whose key touches node_idx as the acted-on device.
        """
        self._q_cache.pop_prefix_node(int(node_idx))

    # -------------------------
    # internal: chunked + cached scoring of candidates
    # -------------------------
    def _batch_score_candidates_with_cache(
        self,
        state_vec: np.ndarray,
        selected_nodes: List[int],
        critic,
        n_types: int,
        D: int,
        E: int,
        A: int,
        max_exploit_candidates: Optional[int] = None,
        chunk_size: int = 1024,
        max_total_candidates: Optional[int] = 200000
    ):
        """
        Chunked and cached candidate Q evaluation.

        - Enumerates candidates for selected_nodes, but will respect max_exploit_candidates
          to limit exploitation indices per node.
        - Uses an LRU cache keyed by (state_key, node_idx, atype, exploit_idx, app_idx).
        - Each cache entry stores (q_value, birth_step); TTL, flush interval, and
          random re-evaluation control reuse.
        """
        # advance cache "time"
        self._cache_step += 1

        # periodic full flush
        if (self._cache_flush_interval is not None and
            self._cache_step > 0 and
            (self._cache_step % self._cache_flush_interval) == 0):
            self._q_cache.clear()

        s_t = torch.tensor(np.asarray(state_vec, dtype=np.float32), dtype=torch.float32,
                           device=self.device).unsqueeze(0)
        with torch.no_grad():
            proj = self.state_proj(s_t).squeeze(0)
        state_key = self._proj_state_key(proj)

        cache_hits: List[Tuple[float, Tuple]] = []
        missing_entries: List[Tuple[Tuple, Tuple]] = []

        total_candidates = 0
        exploit_range_template = range(E)
        if max_exploit_candidates is not None and max_exploit_candidates < E:
            exploit_range_template = range(min(max_exploit_candidates, E))

        # Enumerate candidate keys; pull from cache when safe.
        for node in selected_nodes:
            if getattr(self, "node_dirty", None) is not None and bool(self.node_dirty[node]):
                self._invalidate_node_cache(node)

            for atype in range(n_types):
                for eidx in exploit_range_template:
                    app_idx = 0
                    key = (state_key, int(node), int(atype), int(eidx), int(app_idx))
                    cached = self._q_cache.get(key, None)

                    use_cached = False
                    if cached is not None:
                        # cached may be a raw float (old runs) or (q, birth_step)
                        if isinstance(cached, tuple) and len(cached) == 2:
                            q_val, birth_step = cached
                        else:
                            q_val, birth_step = float(cached), 0

                        # TTL check
                        ttl_ok = True
                        if self._cache_ttl is not None:
                            ttl_ok = (self._cache_step - birth_step) <= self._cache_ttl

                        # random re-sample (spot check)
                        resample = (
                            self._cache_resample_prob > 0.0 and
                            random.random() < self._cache_resample_prob
                        )

                        if ttl_ok and not resample:
                            use_cached = True
                            cache_hits.append(
                                (float(q_val),
                                 (atype,
                                  np.array([eidx], dtype=int),
                                  np.array([node], dtype=int),
                                  app_idx))
                            )
                        else:
                            # treat as stale -> recompute
                            missing_entries.append(
                                (key,
                                 (atype,
                                  np.array([eidx], dtype=int),
                                  np.array([node], dtype=int),
                                  app_idx))
                            )
                    else:
                        # no cached value: schedule for evaluation
                        missing_entries.append(
                            (key,
                             (atype,
                              np.array([eidx], dtype=int),
                              np.array([node], dtype=int),
                              app_idx))
                        )

                    if cached is None or not use_cached:
                        total_candidates += 1
                    if (max_total_candidates is not None and
                        total_candidates >= int(max_total_candidates)):
                        break
                if (max_total_candidates is not None and
                    total_candidates >= int(max_total_candidates)):
                    break
            if (max_total_candidates is not None and
                total_candidates >= int(max_total_candidates)):
                break

        # If nothing is missing, we are done.
        if not missing_entries:
            return cache_hits

        # Process missing entries in chunks.
        idx = 0
        L = len(missing_entries)
        while idx < L:
            end = min(idx + int(chunk_size), L)
            chunk = missing_entries[idx:end]
            action_vecs = []
            cand_actions: List[Tuple[Tuple, Tuple]] = []

            for key, cand in chunk:
                try:
                    vec = self.oracle.encode_action(cand, n_types, D, E, A)
                except Exception:
                    atype, exps, devs, app_idx = cand
                    vec = self.oracle.encode_action(
                        (int(atype),
                         np.array(exps, dtype=int),
                         np.array(devs, dtype=int),
                         int(app_idx)),
                        n_types, D, E, A
                    )
                action_vecs.append(vec)
                cand_actions.append((key, cand))

            try:
                act_t = torch.tensor(np.stack(action_vecs), dtype=torch.float32, device=self.device)
            except Exception:
                action_vecs_np = [np.asarray(v, dtype=np.float32) for v in action_vecs]
                act_t = torch.tensor(np.vstack(action_vecs_np), dtype=torch.float32, device=self.device)

            state_batch = s_t.repeat(len(action_vecs), 1).to(next(critic.parameters()).device)

            with torch.no_grad():
                qvals_t = critic(state_batch, act_t).squeeze(1).cpu().numpy()

            for (key, cand), qv in zip(cand_actions, qvals_t.tolist()):
                qv_f = float(qv)
                # store with birth_step for TTL tracking
                self._q_cache.set(key, (qv_f, self._cache_step))
                cache_hits.append((qv_f, cand))

            del action_vecs, cand_actions, act_t, state_batch, qvals_t
            idx = end

        return cache_hits

    # -------------------------
    # internal: top candidate per node
    # -------------------------
    def _top_candidate_per_node(self, scored_candidates: List[Tuple[float, Tuple]]) -> Dict[int, Tuple[float, Tuple]]:
        """
        scored_candidates: list of (q, (atype, exps, devs, app))
        returns dict node_idx -> (best_q, best_action)
        """
        best: Dict[int, Tuple[float, Tuple]] = {}
        for q, action in scored_candidates:
            _, _, devs, _ = action
            if isinstance(devs, (list, tuple, np.ndarray)):
                if len(devs) == 0:
                    continue
                node = int(devs[0])
            else:
                continue
            cur = best.get(node)
            if cur is None or q > cur[0]:
                best[node] = (q, action)
        return best

    # -------------------------
    # execute()
    # -------------------------
    def execute(self, strat: Strategy, state: np.ndarray, env=None, repair_connected: bool = True):
        env = env or self.env

        meta = (strat.type_mapping.get("meta") if getattr(strat, "type_mapping", None) else None)
        if meta:
            # support either a saved path or a dict of state_dicts
            if isinstance(meta, dict) and "path" in meta:
                try:
                    self.load(meta.get("path"))
                except Exception:
                    pass
            else:
                try:
                    if isinstance(meta.get("state_proj"), dict):
                        self.state_proj.load_state_dict(meta["state_proj"], strict=False)
                    if isinstance(meta.get("node_proj"), dict):
                        self.node_proj.load_state_dict(meta["node_proj"], strict=False)
                    if isinstance(meta.get("struct_feats"), dict):
                        self.struct_feats.load_state_dict(meta["struct_feats"], strict=False)
                except Exception:
                    pass

        # refresh embeddings for dirty nodes
        if self.node_dirty.any():
            if self.node_dirty.sum() > (0.2 * max(1, self.M)):
                self._ensure_all_embeddings(env)
            else:
                dirty_idx = np.nonzero(self.node_dirty)[0].tolist()
                self._recompute_embeddings_indices(env, dirty_idx)
                self.node_dirty[dirty_idx] = False

        v = build_visibility_mask(env, self.role).cpu().numpy()
        #v[:] = 1.0  # uncomment this if you want to make everything visable
        chosen_nodes = set(self.select_devices(env, state, visible_mask=v, k=self.select_k))

        # allowed nodes are visible AND in chosen_nodes
        allowed_nodes = set(int(i) for i in np.where((v > 0.5))[0] if int(i) in chosen_nodes)

        # If no allowed nodes, fallback to low_policy behavior (keeps legacy)
        if not allowed_nodes:
            print(f"[MetaDOAR] Warning: α={self.alpha}, select_k={self.select_k}, no allowed nodes — fallback triggered.")
            groups_raw = self.low_policy.select_action(state, env=env)
            filtered = []
            for (atype, exps, devs, app_idx) in groups_raw:
                try:
                    devs_list = list(devs)
                except Exception:
                    devs_list = devs if isinstance(devs, (list, tuple, np.ndarray)) else []
                devs_filtered = np.array([int(d) for d in devs_list if int(d) in allowed_nodes], dtype=int)
                if devs_filtered.size == 0:
                    continue
                filtered.append((atype, exps, devs_filtered, app_idx))
            if not filtered:
                if allowed_nodes:
                    d0 = next(iter(allowed_nodes))
                    filtered = [(0, np.array([0], dtype=int), np.array([d0], dtype=int), 0)]
                else:
                    filtered = [(0, np.array([0], dtype=int), np.array([], dtype=int), 0)]
            # mark k-hop neighbors dirty and store last selection
            try:
                A = adjacency_matrix_from_env(env)
                neighbors = self._get_k_hop_nodes(A, list(chosen_nodes), k=self.k_hop_invalidate)
                for n in neighbors:
                    if 0 <= n < self.M:
                        self.node_dirty[n] = True
                        self._invalidate_node_cache(n)
            except Exception:
                pass
            self._last_selection = {
                "state": np.asarray(state, dtype=np.float32),
                "selected": np.array(sorted(chosen_nodes), dtype=np.int32)
            }
            return filtered

        # Otherwise: use cache + batched critic evaluation to score candidate actions restricted to allowed_nodes
        D = int(getattr(self.oracle, "D_init", getattr(self.oracle, "Max_network_size", 0)))
        E = int(getattr(self.oracle, "E_init", getattr(self.oracle, "MaxExploits", 0)))
        A = int(getattr(self.oracle, "A_init", getattr(self.oracle, "get_num_app_indices", lambda: 0)()))

        # pick ddpg critic depending on role
        if self.role == "defender":
            ddpg = self.oracle.defender_ddpg
            n_types = self.oracle.n_def_types
        else:
            ddpg = self.oracle.attacker_ddpg
            n_types = self.oracle.n_att_types

        selected_nodes = sorted(list(allowed_nodes))
        scored = self._batch_score_candidates_with_cache(
            state_vec=state,
            selected_nodes=selected_nodes,
            critic=ddpg['critic'],
            n_types=n_types, D=D, E=E, A=A,
            max_exploit_candidates=1,    # keep small candidate set per node
            chunk_size=512,
            max_total_candidates=1000
        )

        best_map = self._top_candidate_per_node(scored)
        filtered = []
        for node in sorted(best_map.keys()):
            q, action = best_map[node]
            atype, exps, devs, app_idx = action
            filtered.append((int(atype), np.array(exps, dtype=int), np.array(devs, dtype=int), int(app_idx)))

        # fallback if no filtered (shouldn't normally happen)
        if not filtered:
            if allowed_nodes:
                d0 = next(iter(allowed_nodes))
                filtered = [(0, np.array([0], dtype=int), np.array([d0], dtype=int), 0)]
            else:
                filtered = [(0, np.array([0], dtype=int), np.array([], dtype=int), 0)]

        # mark k-hop neighbors dirty + invalidate cache entries touching them
        try:
            A = adjacency_matrix_from_env(env)
            neighbors = self._get_k_hop_nodes(A, list(chosen_nodes), k=self.k_hop_invalidate)
            for n in neighbors:
                if 0 <= n < self.M:
                    self.node_dirty[n] = True
                    self._invalidate_node_cache(n)
        except Exception:
            pass

        self._last_selection = {
            "state": np.asarray(state, dtype=np.float32),
            "selected": np.array(sorted(chosen_nodes), dtype=np.int32)
        }
        return filtered

    # -------------------------
    # bfs helper
    # -------------------------
    def _get_k_hop_nodes(self, A: torch.Tensor, seed_nodes: List[int], k: int = 1) -> List[int]:
        if len(seed_nodes) == 0:
            return []

        # Build adjacency lists without converting to a huge dense matrix if sparse
        if getattr(A, "is_sparse", False):
            co = A.coalesce()
            inds = co.indices().cpu().numpy()
            rows = inds[0]; cols = inds[1]
            neighbors_map: Dict[int, List[int]] = {int(i): [] for i in range(self.M)}
            for u, v in zip(rows.tolist(), cols.tolist()):
                if v not in neighbors_map[u]:
                    neighbors_map[u].append(v)
        else:
            A_np = A.cpu().numpy()
            neighbors_map = {int(i): np.nonzero(A_np[i])[0].tolist() for i in range(A_np.shape[0])}

        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(int(k)):
            next_frontier = set()
            for u in list(frontier):
                nbrs = neighbors_map.get(int(u), [])
                for v in nbrs:
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
            if not next_frontier:
                break
            frontier = next_frontier
        return sorted(visited)

    # -------------------------
    # replay / controller training
    # -------------------------
    def store_transition(self, reward: float, next_state: Optional[np.ndarray] = None, done: bool = False):
        if not hasattr(self, "_last_selection"):
            return
        st = self._last_selection["state"]
        sel = self._last_selection["selected"]
        ns = np.asarray(next_state, dtype=np.float32) if next_state is not None else None
        self.replay.append((st, sel, float(reward), ns, bool(done)))
        delattr(self, "_last_selection")

    def _batch_sample(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(self.replay, self.batch_size)
        return batch

    def train_controller(self, iterations: int = 1):
        """
        Train the internal projector from replay buffer (lightweight MSE regression).
        Returns total loss.
        """
        if len(self.replay) == 0:
            return 0.0
        total_loss = 0.0
        for it in range(int(iterations)):
            batch = self._batch_sample()
            if batch is None:
                break
            states = []
            sel_masks = []
            targets = []
            for st, sel, rew, ns, done in batch:
                states.append(st)
                targets.append(rew)
                mask = np.zeros((self.M,), dtype=np.float32)
                if sel is not None and len(sel) > 0:
                    mask[sel] = 1.0 / float(len(sel))
                sel_masks.append(mask)
            states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
            masks_t = torch.tensor(np.stack(sel_masks), dtype=torch.float32, device=self.device)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=self.device).unsqueeze(1)

            proj_t = self.state_proj(states_t)
            E = self.E_cache.detach()
            scores_nodes = (E @ proj_t.t()).t()
            scores_nodes = scores_nodes + float(self.node_bias.detach().cpu().item())
            preds = (scores_nodes * masks_t).sum(dim=1, keepdim=True)

            loss = F.mse_loss(preds, targets_t)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.state_proj.parameters()) + list(self.node_proj.parameters()), 1.0)
            self.opt.step()
            total_loss += float(loss.item())
            self._train_steps += 1
            if (self._train_steps % 100) == 0:
                tau = 0.02
                with torch.no_grad():
                    for tgt_p, src_p in zip(self.target_state_proj.parameters(), self.state_proj.parameters()):
                        tgt_p.data.copy_(tau * src_p.data + (1 - tau) * tgt_p.data)
        return total_loss


    # -------------------------
    # DO-compatible train(...) wrapper
    # -------------------------
    def train(self,
              opponent_strategies: List[Strategy],
              opponent_equilibrium: List[float],
              T: int = 15000,
              σ: float = 1.0,
              σ_min: float = 1e-5,
              return_meta: bool = False,
              **kwargs) -> Strategy:
        """
        DO-compatible signature. By default delegates to oracle.ddpg_best_response()
        which returns a Strategy (actor/critic/state-dicts).

        If return_meta=True, we instead save the current meta-controller to disk
        and return a Strategy with type_mapping={'meta': {'path': <saved path>}}.

        When return_meta=False, we call ddpg_best_response with meta_controller=self
        so that the MetaDOAR controller is trained in observer mode during BR rollout.
        """
        eq = list(opponent_equilibrium) if opponent_equilibrium is not None else [
            1.0 / max(1, len(opponent_strategies))
        ] * max(1, len(opponent_strategies))

        # Pure "export meta" mode: just snapshot current meta-params as a Strategy.
        if return_meta:
            fd, path = tempfile.mkdtemp(prefix=f"meta_{self.role}_")
            save_path = os.path.join(path, "meta.pth")
            self.save(save_path)
            meta_payload = {"path": save_path}
            return Strategy(type_mapping={"meta": meta_payload})

        # Normal BR training: train DOAR while MetaDOAR observes and learns via meta_controller=self
        try:
            new_strat = self.oracle.ddpg_best_response(
                opponent_strategies,
                eq,                   # or opponent_equilibrium, they are equivalent here
                self.role,
                training_steps=int(T),
                σ=σ,
                σ_min=σ_min,
                meta_controller=self,
                **kwargs,             # e.g., exploit_override if you pass it
            )
            return new_strat
        except Exception:
            # Fallback: random single-device strategy if BR fails
            try:
                D = self.oracle.D_init
                acts = [
                    (0,
                     np.array([], dtype=int),
                     np.array([np.random.randint(0, D)], dtype=int),
                     0)
                    for _ in range(getattr(self.oracle, "steps_per_episode", 50))
                ]
                return Strategy(actions=acts)
            except Exception:
                raise


    # -------------------------
    # debug helper
    # -------------------------
    def debug_scores(self, env, state_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.node_dirty.any():
            self._ensure_all_embeddings(env)
        s_t = torch.tensor(np.asarray(state_vec, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            proj = self.state_proj(s_t).squeeze(0)
            scores = (self.E_cache @ proj).cpu().numpy().astype(float) + float(self.node_bias.detach().cpu().item())
        idx = np.arange(self.M)
        return idx, scores
