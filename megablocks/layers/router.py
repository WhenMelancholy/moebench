import torch
import torch.nn as nn

from megablocks.layers import common
from megablocks.layers.arguments import Arguments


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedRouter(torch.nn.Module):

    def __init__(
        self,
        args: Arguments,
        bias_u=None,
        bias_update_step=None,
    ):
        super().__init__()
        self.args = args

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device,
        )
        args.init_method(self.layer.weight)

        self.bias_u = bias_u
        if self.bias_u:
            self.bias = nn.Parameter(
                torch.zeros(args.moe_num_experts),
                requires_grad=False,
            )

            assert (
                bias_update_step is not None
            ), "Gradient accumulation steps must be provided for bias update."
            self.ci_buffer = None
            self.bias_update_step = bias_update_step
            self.accum_steps = 0

    def jitter(self, x):
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        if self.bias_u is not None:
            _, selected_experts = torch.topk(
                scores + self.bias.unsqueeze(0),
                self.args.moe_top_k,
                dim=-1,
            )
            scores = scores.gather(-1, selected_experts)
            return scores, selected_experts
        if self.args.moe_top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.args.moe_top_k, dim=-1)

    def forward(
        self,
        x,
        random_router: bool = False,
        prune_list: torch.Tensor | None = None,
    ):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        if self.args.moe_expert_choice:
            # Get probability for each token
            bs, sq, _ = x.shape
            capacity = (
                self.args.moe_top_k
            )  # Use top k as the capacity to match regular MoEs
            logits = self.layer(x)
            if random_router:
                logits = torch.randn_like(logits)
            if prune_list is not None:
                prune_list = prune_list.to(logits.device)
                logits[..., prune_list] = float("-inf")
            scores = logits.softmax(dim=-1)  # [batch_size, seq_len, num_experts]
            if self.bias_u != None:
                raise NotImplementedError(
                    "Bias update is not implemented for expert choice"
                )
            expert_weights, expert_indices = torch.topk(
                scores.transpose(1, 2),
                (capacity * sq) // self.args.moe_num_experts,
                dim=-1,
            )  # [batch_size, num_experts, k]
        elif self.args.moe_expert_choice_grouped:
            bs, sq, _ = x.shape
            capacity = (
                self.args.moe_top_k
            )  # Use top k as the capacity to match regular MoEs
            logits = self.layer(x.view(-1, x.shape[-1]))  # [bs & sq, num_experts]
            if random_router:
                logits = torch.randn_like(logits)
            if prune_list is not None:
                prune_list = prune_list.to(logits.device)
                logits[..., prune_list] = float("-inf")
            scores = logits.softmax(dim=-1)
            if self.bias_u != None:
                raise NotImplementedError(
                    "Bias update is not implemented for expert choice grouped"
                )
            expert_weights, expert_indices = torch.topk(
                scores.transpose(0, 1),
                (capacity * bs * sq) // self.args.moe_num_experts,
                dim=-1,
            )  # [num_experts, k]
        else:
            logits = self.layer(x.view(-1, x.shape[-1]))
            if random_router:
                logits = torch.randn_like(logits)
            if prune_list is not None:
                prune_list = prune_list.to(logits.device)
                logits[..., prune_list] = float("-inf")
            scores = logits.softmax(dim=-1)
            expert_weights, expert_indices = self._top_k(scores)

        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        # ========================= 分布式更新 bias =========================
        if self.bias_u != None and self.training:
            local_ci = torch.bincount(
                expert_indices.flatten(),
                minlength=self.args.moe_num_experts,
            ).float()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    local_ci,
                    op=torch.distributed.ReduceOp.SUM,
                )
            if self.ci_buffer is None:
                self.ci_buffer = local_ci
            else:
                self.ci_buffer += local_ci
            self.accum_steps += 1
            if self.accum_steps == self.bias_update_step:
                aggregated_ci = self.ci_buffer
                global_mean = aggregated_ci.mean()
                delta_bias = (global_mean - aggregated_ci).sign()
                self.bias.data = self.bias.data + self.bias_u * delta_bias
                self.ci_buffer = None
                self.accum_steps = 0
        # ==================================================================

        expert_indices = (
            _uniform_expert_assignment(expert_indices, self.args.moe_num_experts)
            if self.args.uniform_expert_assignment
            else expert_indices
        )
        return scores, logits, expert_weights, expert_indices
