"""
RLTT-specific algorithms for verl.

This module implements the core RLTT algorithms including:
- Multi-loop log-probability aggregation
- Loop weighting strategies (uniform, progressive, exit_pdf, learned)
- RLTT policy gradient loss
"""

import torch
from torch import nn
from typing import List, Optional, Literal, Dict
import verl.utils.torch_functional as verl_F


class LearnedLoopWeights(nn.Module):
    """Learnable loop weights for RLTT in verl.

    This module maintains trainable logits that are converted to normalized
    weights via softmax. The weights are jointly optimized with the policy.
    """

    def __init__(
        self,
        num_loops: int,
        init_strategy: str = "uniform",
        init_alpha: float = 1.0,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize learned loop weights.

        Args:
            num_loops: Number of loops (T_max)
            init_strategy: How to initialize weights ("uniform" or "progressive")
            init_alpha: Alpha for progressive initialization
            temperature: Softmax temperature (higher = more uniform)
            device: Device to place weights on
        """
        super().__init__()
        self.num_loops = num_loops
        self.temperature = temperature

        # Initialize logits based on strategy
        if init_strategy == "uniform":
            init_logits = torch.zeros(num_loops, device=device)
        elif init_strategy == "progressive":
            t = torch.arange(1, num_loops + 1, dtype=torch.float32, device=device)
            target_weights = t ** init_alpha
            target_weights = target_weights / target_weights.sum()
            init_logits = torch.log(target_weights + 1e-8)
        else:
            init_logits = torch.zeros(num_loops, device=device)

        self.logits = nn.Parameter(init_logits)

    def forward(self) -> torch.Tensor:
        """Get normalized weights via softmax."""
        return torch.softmax(self.logits / self.temperature, dim=0)

    def get_weights_for_logging(self) -> Dict[str, float]:
        """Get current weights as a dict for logging."""
        with torch.no_grad():
            weights = self.forward()
            return {f"loop_{t+1}_weight": w.item() for t, w in enumerate(weights)}


def compute_loop_weights(
    num_loops: int,
    strategy: Literal["uniform", "progressive", "exit_pdf", "learned"] = "uniform",
    alpha: float = 1.0,
    exit_pdf: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute loop weights for RLTT loss.

    Args:
        num_loops: Number of recurrent loops (T_max)
        strategy: Weighting strategy
            - "uniform": ω_t = 1/T_max (Option A from paper)
            - "progressive": ω_t = t^α / Σ s^α (Option B from paper)
            - "exit_pdf": Use model's native exit probability distribution
            - "learned": Return uniform as placeholder (actual weights from LearnedLoopWeights)
        alpha: Exponent for progressive weighting (default: 1.0 for linear)
        exit_pdf: Pre-computed exit PDF tensor of shape (batch_size, seq_len, num_loops)
                  or (num_loops,) for static weights. Required if strategy="exit_pdf"
        device: Device to place weights on

    Returns:
        weights: Tensor of shape (num_loops,) for uniform/progressive/learned,
                 or the exit_pdf tensor for exit_pdf strategy
    """
    if strategy == "uniform":
        weights = torch.ones(num_loops, device=device) / num_loops

    elif strategy == "progressive":
        # Progressive weighting: ω_t = t^α / Σ s^α
        # Loop indices are 1-indexed: t ∈ {1, 2, ..., T_max}
        loop_indices = torch.arange(1, num_loops + 1, dtype=torch.float32, device=device)
        unnormalized = loop_indices ** alpha
        weights = unnormalized / unnormalized.sum()

    elif strategy == "exit_pdf":
        if exit_pdf is None:
            raise ValueError("exit_pdf tensor required for exit_pdf strategy")
        weights = exit_pdf

    elif strategy == "learned":
        # Return uniform as placeholder; actual weights come from LearnedLoopWeights module
        weights = torch.ones(num_loops, device=device) / num_loops

    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    return weights


def aggregate_multi_loop_log_probs(
    per_loop_log_probs: List[torch.Tensor],
    weights: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Aggregate log-probabilities across loops using weighted sum.

    Implements: log P_RLTT(y|x) = Σ_t ω_t * log P^(t)(y|x)

    Args:
        per_loop_log_probs: List of T tensors, each of shape (batch_size, response_length)
                           containing log P^(t)(y_j | x, y_{<j}) for each loop t
        weights: Loop weights of shape (num_loops,) or (batch_size, response_length, num_loops)
        response_mask: Optional mask of shape (batch_size, response_length)

    Returns:
        aggregated_log_probs: Tensor of shape (batch_size, response_length)
    """
    num_loops = len(per_loop_log_probs)

    # Stack to (batch_size, response_length, num_loops)
    stacked_log_probs = torch.stack(per_loop_log_probs, dim=-1)

    # Handle different weight shapes
    if weights.dim() == 1:
        # Static weights: (num_loops,) -> broadcast to all positions
        aggregated = (stacked_log_probs * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    elif weights.dim() == 3:
        # Dynamic weights: (batch_size, response_length, num_loops)
        aggregated = (stacked_log_probs * weights).sum(dim=-1)
    else:
        raise ValueError(f"Unexpected weights shape: {weights.shape}")

    return aggregated


def compute_rltt_policy_loss(
    per_loop_log_probs: List[torch.Tensor],
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    loop_weights: torch.Tensor,
    num_generations: int = 8,
) -> tuple:
    """Compute RLTT policy gradient loss with multi-loop aggregation.

    This implements the pure REINFORCE-style RLTT objective from the paper (Equation 6):
        J_RLTT_PG(θ) = -E[ (1/g) Σ_i (1/|y_i|) Σ_j Σ_t ω_t * log P^(t)(y_{i,j}) * Â_i ]

    This is purely ON-POLICY with NO probability ratios (π_θ/π_old).
    The advantage Â_i is computed via GRPO-style group normalization.

    Args:
        per_loop_log_probs: List of T tensors for current policy per-loop log-probs,
                           each of shape (bs, response_length)
        advantages: Advantage estimates, shape (bs, response_length)
        eos_mask: Response mask, shape (bs, response_length)
        loop_weights: RLTT loop weights, shape (num_loops,) or (bs, response_length, num_loops)
        num_generations: Number of generations per prompt for GRPO-style normalization

    Returns:
        pg_loss: Policy gradient loss (scalar)
        rltt_metrics: Dict with RLTT-specific metrics
    """
    # Aggregate current policy log-probs across loops (Equation 6 inner sum over t)
    # rltt_log_prob = Σ_t ω_t * log P^(t)(y_j | x, y_{<j})
    rltt_log_prob = aggregate_multi_loop_log_probs(
        per_loop_log_probs, loop_weights, eos_mask
    )

    # Pure REINFORCE policy gradient: -advantage * log_prob
    # This is the key difference from PPO/GRPO which uses: -advantage * (π_θ/π_old)
    #
    # From paper Equation 6:
    # J_RLTT_PG(θ) = -E[ (1/g) Σ_i (1/|y_i|) Σ_j Σ_t ω_t * log P^(t)(y_{i,j}) * Â_i ]
    #
    # The masked_mean handles the (1/|y_i|) Σ_j part
    # The advantages already have GRPO-style normalization applied

    # Compute per-token policy gradient loss
    pg_losses = -advantages * rltt_log_prob

    # Average over tokens (masked) - this gives us the sequence-level loss
    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    # Compute per-loop metrics for logging
    rltt_metrics = {}
    for t, loop_lp in enumerate(per_loop_log_probs):
        loop_mean_lp = verl_F.masked_mean(loop_lp, eos_mask).item()
        rltt_metrics[f'rltt/loop_{t+1}_log_prob'] = loop_mean_lp

    # Log aggregated log-prob
    rltt_metrics['rltt/aggregated_log_prob'] = verl_F.masked_mean(rltt_log_prob, eos_mask).item()

    # Log weight distribution if static
    if loop_weights.dim() == 1:
        for t, w in enumerate(loop_weights.tolist()):
            rltt_metrics[f'rltt/loop_{t+1}_weight'] = w
    elif loop_weights.dim() == 3:
        # Dynamic exit_pdf weights: log mean exit probability per loop
        for t in range(loop_weights.shape[-1]):
            rltt_metrics[f'rltt/exit_prob_loop_{t+1}_mean'] = loop_weights[:, :, t].mean().item()

    return pg_loss, rltt_metrics


def compute_rltt_kl_penalty(
    per_loop_log_probs: List[torch.Tensor],
    per_loop_ref_log_probs: List[torch.Tensor],
    loop_weights: torch.Tensor,
    kl_penalty_type: str = "kl",
) -> torch.Tensor:
    """Compute KL penalty for RLTT with multi-loop aggregation.

    Args:
        per_loop_log_probs: Current policy per-loop log-probs
        per_loop_ref_log_probs: Reference policy per-loop log-probs
        loop_weights: RLTT loop weights
        kl_penalty_type: Type of KL penalty ('kl', 'abs', 'mse', 'low_var_kl')

    Returns:
        kl_penalty: Per-token KL penalty, shape (bs, response_length)
    """
    # Aggregate log-probs
    rltt_log_prob = aggregate_multi_loop_log_probs(per_loop_log_probs, loop_weights)
    rltt_ref_log_prob = aggregate_multi_loop_log_probs(per_loop_ref_log_probs, loop_weights)

    if kl_penalty_type == "kl":
        return rltt_log_prob - rltt_ref_log_prob
    elif kl_penalty_type == "abs":
        return (rltt_log_prob - rltt_ref_log_prob).abs()
    elif kl_penalty_type == "mse":
        return 0.5 * (rltt_log_prob - rltt_ref_log_prob).square()
    elif kl_penalty_type == "low_var_kl":
        kl = rltt_ref_log_prob - rltt_log_prob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    else:
        raise NotImplementedError(f"KL penalty type '{kl_penalty_type}' not implemented")
