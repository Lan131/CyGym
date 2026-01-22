# strategy.py
from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Type
import logging

import numpy as np
import torch
import torch.nn as nn


def _clone_state_dict(sd: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Local, cycle-free clone util: detach, clone, move to CPU for each tensor.
    Returns None if sd is None.
    """
    if sd is None:
        return None
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
    return out


class Strategy:
    """
    Lightweight carrier for either:
      • a parametric strategy (stores actor/critic state_dicts), or
      • a fixed-action/baseline strategy (stores actions or baseline_name), or
      • a typed strategy (stores type_mapping for hierarchical/meta/committee/IPPO/etc.)

    This file intentionally has **no** imports from your DO agent to avoid cycles.
    Other modules (do_agent, hierarchical_br, meta_hierarchical, …) should import Strategy from here.
    """

    def __init__(
        self,
        actor_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        critic_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        actions: Optional[List[Tuple]] = None,
        baseline_name: Optional[str] = None,
        actor_dims: Optional[Tuple[int, int]] = None,   # (state_dim, action_dim) to rebuild Actor
        critic_dims: Optional[Tuple[int, int]] = None,  # kept for compatibility; not strictly needed
        type_mapping: Optional[Dict] = None,            # e.g., {"hierarchical": {...}}, {"meta": {...}}, {"committee": obj}, ...
    ):
        # Parametric iff both nets are present
        self.actor_state_dict: Optional[Dict[str, torch.Tensor]] = actor_state_dict
        self.critic_state_dict: Optional[Dict[str, torch.Tensor]] = critic_state_dict

        # Non-parametric flavors
        self.actions: Optional[List[Tuple]] = actions
        self.baseline_name: Optional[str] = baseline_name
        self.type_mapping: Dict = type_mapping or {}

        # Shapes (when available, use them to rebuild networks exactly)
        self.actor_dims: Optional[Tuple[int, int]] = actor_dims
        self.critic_dims: Optional[Tuple[int, int]] = critic_dims

        # Optional running stats
        self.payoffs: List[float] = []

    # -------------------- serialization for multiprocessing --------------------

    def to_payload(self) -> dict:
        """
        Lightweight, pickle-friendly snapshot of this strategy.
        NOTE: type_mapping must itself be picklable if you plan to ship it to workers.
        """
        return {
            "baseline_name":     self.baseline_name,
            "actions":           self.actions,
            "actor_state_dict":  _clone_state_dict(self.actor_state_dict),
            "critic_state_dict": _clone_state_dict(self.critic_state_dict),
            "actor_dims":        self.actor_dims,
            "critic_dims":       self.critic_dims,
            "type_mapping":      self.type_mapping,
        }

    @classmethod
    def from_payload(cls, p: dict) -> "Strategy":
        """
        Rebuild a Strategy from a payload produced by to_payload().
        """
        return cls(
            actor_state_dict  = p.get("actor_state_dict"),
            critic_state_dict = p.get("critic_state_dict"),
            actions           = p.get("actions"),
            baseline_name     = p.get("baseline_name"),
            actor_dims        = p.get("actor_dims"),
            critic_dims       = p.get("critic_dims"),
            type_mapping      = p.get("type_mapping"),
        )

    # -------------------- simple helpers --------------------

    def is_parametric(self) -> bool:
        """True iff this strategy stores both actor and critic weights."""
        return (self.actor_state_dict is not None) and (self.critic_state_dict is not None)

    def add_payoff(self, payoff: float) -> None:
        self.payoffs.append(float(payoff))

    def average_payoff(self) -> float:
        return float(np.mean(self.payoffs)) if self.payoffs else 0.0

    # -------------------- model rehydration --------------------

    def load_actor(
        self,
        ActorClass: Type[nn.Module],
        *args,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Optional[nn.Module]:
        """
        Rebuild exactly the same Actor that was saved, inferring its input/output sizes from either
        self.actor_dims (preferred) or from weight shapes (fc1.weight, fc3.weight) in the checkpoint.

        Returns a model placed on `device` and set to eval(), or None if this strategy has no actor.
        """
        if self.actor_state_dict is None:
            return None

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer recorded dims
        if self.actor_dims:
            state_dim, action_dim = self.actor_dims
            logging.debug(
                "[Strategy.load_actor] using recorded actor_dims: in=%d, out=%d",
                state_dim, action_dim
            )
        else:
            # Infer from checkpoint shapes:
            try:
                w1 = self.actor_state_dict["fc1.weight"]  # [hidden, state_dim]
                w3 = self.actor_state_dict["fc3.weight"]  # [action_dim, hidden]
            except KeyError as e:
                raise KeyError(f"Actor checkpoint missing expected key: {e}") from e
            _, state_dim = w1.shape
            action_dim, _ = w3.shape
            logging.debug(
                "[Strategy.load_actor] inferred dims from ckpt: in=%d, out=%d",
                state_dim, action_dim
            )

        actor = ActorClass(state_dim, action_dim, seed, device)
        try:
            actor.load_state_dict(self.actor_state_dict)
        except RuntimeError as e:
            logging.error(f"[Strategy.load_actor] load_state_dict failed: {e}")
            raise

        actor = actor.to(device).eval()
        for p in actor.parameters():
            p.requires_grad = False
        return actor

    def load_critic(
        self,
        CriticClass: Type[nn.Module],
        *args,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> Optional[nn.Module]:
        """
        Rebuild the Critic from its checkpoint. We split the first layer's input into (state_dim, action_dim).
        Priority for the split:
          1) self.actor_dims if present (preferred — ensures exact split),
          2) otherwise, fallback to an even split as last resort.
        """
        if self.critic_state_dict is None:
            return None

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Peek at total input size from first layer
        try:
            w1 = self.critic_state_dict["fc1.weight"]  # [hidden, state_dim + action_dim]
        except KeyError as e:
            raise KeyError(f"Critic checkpoint missing expected key: {e}") from e

        _, total_in = w1.shape

        if self.actor_dims:
            state_dim, action_dim = self.actor_dims
            if state_dim + action_dim != total_in:
                logging.warning(
                    "[Strategy.load_critic] actor_dims sum (%d + %d) != critic fc1 in-features (%d). "
                    "Overriding action_dim to match checkpoint.",
                    state_dim, action_dim, total_in
                )
                action_dim = total_in - state_dim
        else:
            # Fallback: split evenly
            state_dim = total_in // 2
            action_dim = total_in - state_dim
            logging.warning(
                "[Strategy.load_critic] actor_dims unavailable; inferred state_dim=%d, action_dim=%d from total_in=%d",
                state_dim, action_dim, total_in
            )

        critic = CriticClass(state_dim, action_dim, seed, device)
        try:
            critic.load_state_dict(self.critic_state_dict)
        except RuntimeError as e:
            logging.error(f"[Strategy.load_critic] load_state_dict failed: {e}")
            raise

        critic = critic.to(device).eval()
        for p in critic.parameters():
            p.requires_grad = False
        return critic

    # -------------------- repr --------------------

    def __repr__(self) -> str:
        kind = (
            "Parametric"
            if self.is_parametric()
            else ("Baseline" if self.baseline_name is not None else
                  ("FixedSeq" if self.actions is not None else
                   ("Typed" if self.type_mapping else "Empty")))
        )
        return f"<Strategy {kind} avg_payoff={self.average_payoff():.3f}>"
