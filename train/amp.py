"""
Adversarial Motion Priors (AMP): discriminator on state transitions (s, s').
Used to reward the policy for producing motions similar to reference (expert) data.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def load_reference_transitions(
    expert_path: str | Path,
    obs_dim: int,
    max_transitions: Optional[int] = None,
) -> np.ndarray:
    """
    Load (s, s') pairs from expert .npz (list of obs arrays).
    Returns array of shape (N, 2, obs_dim): N transition pairs.
    """
    data = np.load(str(expert_path), allow_pickle=True)
    obs_list = data["obs"]
    if hasattr(obs_list, "tolist"):
        obs_list = obs_list.tolist()
    if not isinstance(obs_list, list):
        obs_list = list(obs_list)

    transitions = []
    for traj in obs_list:
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        if traj.shape[0] < 2 or traj.shape[1] != obs_dim:
            continue
        for i in range(traj.shape[0] - 1):
            transitions.append(np.stack([traj[i], traj[i + 1]], axis=0))
            if max_transitions and len(transitions) >= max_transitions:
                return np.stack(transitions, axis=0)

    if not transitions:
        raise ValueError(f"No valid transitions in {expert_path} (obs_dim={obs_dim})")
    return np.stack(transitions, axis=0)


class AMPDiscriminator(nn.Module):
    """
    MLP that takes concatenated (s, s') and outputs a scalar logit.
    Expert transitions should score near +1, policy near -1 (LSGAN).
    Style reward: max(0, 1 - 0.25 * (D(s,s') - 1)^2).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        input_dim = obs_dim * 2  # (s, s')
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Logit for transition (s, s'). Shape (batch,) or (batch, 1)."""
        x = torch.cat([s, s_next], dim=-1)
        return self.net(x).squeeze(-1)

    def predict_reward(self, s: np.ndarray, s_next: np.ndarray) -> np.ndarray:
        """
        Style reward for policy: max(0, 1 - 0.25*(D(s,s') - 1)^2).
        Higher when transition looks like expert (D near 1).
        """
        with torch.no_grad():
            t_s = torch.as_tensor(s, dtype=torch.float32, device=next(self.parameters()).device)
            t_s_next = torch.as_tensor(s_next, dtype=torch.float32, device=next(self.parameters()).device)
            if t_s.dim() == 1:
                t_s = t_s.unsqueeze(0)
                t_s_next = t_s_next.unsqueeze(0)
            logit = self.forward(t_s, t_s_next)
            d = torch.tanh(logit)  # bound to [-1, 1]; we want expert=1, policy=-1
            # reward = max(0, 1 - 0.25*(d - 1)^2)
            r = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
            return r.cpu().numpy()


class AMPTrainer:
    """Train the AMP discriminator on expert vs policy transitions (LSGAN-style)."""

    def __init__(
        self,
        discriminator: AMPDiscriminator,
        expert_transitions: np.ndarray,
        lr: float = 1e-4,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        self.disc = discriminator
        self.expert = torch.as_tensor(expert_transitions, dtype=torch.float32)  # (N, 2, obs_dim)
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.disc.to(self.device)
        self.expert = self.expert.to(self.device)
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        self._expert_idx = np.arange(len(self.expert))

    def update(
        self,
        policy_s: np.ndarray,
        policy_s_next: np.ndarray,
    ) -> dict[str, float]:
        """
        One update step: sample batches of expert and policy transitions, train D.
        policy_s, policy_s_next: (M, obs_dim) from rollout.
        """
        self.disc.train()
        n_exp = len(self.expert)
        n_pol = len(policy_s)
        if n_exp == 0 or n_pol == 0:
            return {"amp_d_loss": 0.0}

        # Sample
        exp_idx = np.random.choice(n_exp, size=min(self.batch_size, n_exp), replace=True)
        pol_idx = np.random.choice(n_pol, size=min(self.batch_size, n_pol), replace=True)

        exp_batch = self.expert[exp_idx]  # (B, 2, obs_dim)
        s_exp = exp_batch[:, 0]
        s_next_exp = exp_batch[:, 1]

        s_pol = torch.as_tensor(policy_s[pol_idx], dtype=torch.float32, device=self.device)
        s_next_pol = torch.as_tensor(policy_s_next[pol_idx], dtype=torch.float32, device=self.device)

        # LSGAN: D(real)->1, D(fake)->-1
        d_real = self.disc(s_exp, s_next_exp)
        d_fake = self.disc(s_pol, s_next_pol)
        loss_real = ((d_real - 1) ** 2).mean()
        loss_fake = ((d_fake + 1) ** 2).mean()
        loss_d = 0.5 * (loss_real + loss_fake)

        self.opt.zero_grad()
        loss_d.backward()
        self.opt.step()

        return {"amp_d_loss": loss_d.item()}
