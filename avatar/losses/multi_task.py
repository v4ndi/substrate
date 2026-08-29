import random

import torch
import torch.nn as nn


class WeightLossBalancer(nn.Module):
    def __init__(self, loss_weights: dict[str, float]):
        super().__init__()
        self.loss_weights = nn.ParameterDict({
            name: nn.Parameter(
                torch.tensor(value, dtype=torch.float32), requires_grad=False
            )
            for name, value in loss_weights.items()
        })

    def forward(self, losses_name, losses_value):
        total_loss = 0.0
        for loss_name, loss_value in zip(losses_name, losses_value, strict=False):
            if loss_name not in self.loss_weights:
                raise ValueError(f"Loss {loss_name} not found in weights")
            total_loss += self.loss_weights[loss_name] * loss_value
        return total_loss


class UncertainlyLossBalancer(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        self.loss_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses_name, losses_value):
        losses_value = list(losses_value)
        total_loss = 0.0
        for i, loss_value in enumerate(losses_value):
            loss_value += 1e-8
            log_var = torch.clamp(self.loss_vars[i], -10, 10)
            weighted = torch.exp(-log_var) * loss_value + log_var

            total_loss += weighted
        print(f"weights: {torch.exp(-self.loss_vars.detach())}")
        print(total_loss)
        return total_loss


class GradNormLossBalancer(nn.Module):
    def __init__(self, num_losses, alpha=1.5, renormilize_weights=False):
        super().__init__()
        self.alpha = alpha
        self.log_weights = nn.Parameter(torch.zeros(num_losses))
        # self.weights = nn.Parameter(torch.ones(num_losses))
        self.num_losses = num_losses
        self.initial_losses = None
        self.renormilize_weights = renormilize_weights

    def forward(self, losses, shared_representation):
        losses = list(losses)
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [loss.detach().item() for loss in losses], device=losses[0].device
            )
        weights = torch.exp(self.log_weights)
        if self.renormilize_weights:
            weights = weights * (self.num_losses / weights.sum())
        weighted_loss = [w * loss for w, loss in zip(weights, losses, strict=False)]
        total_loss = sum(weighted_loss)

        if not torch.is_grad_enabled():
            return total_loss, 0.0

        grad_norms = []
        for wl in weighted_loss:
            if wl.detach().item() == 0.0:
                grad_norms.append(torch.tensor(0.0, device=wl.device))
            else:
                g = torch.autograd.grad(
                    wl,
                    shared_representation,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if g is None:
                    print("asdsad")
                    grad_norms.append(torch.tensor(0.0, device=wl.device))
                else:
                    grad_norms.append(g.norm() + 1e-6)
        grad_norms = torch.stack(grad_norms)
        loss_ratios = (
            torch.tensor([loss.item() for loss in losses], device=losses[0].device)
            / self.initial_losses
        )

        inverse_train_rates = loss_ratios / loss_ratios.mean()
        target_grads = grad_norms.mean() * (inverse_train_rates**self.alpha)
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grads.detach()))
        print(f"grad_norms: {grad_norms.detach()}")
        print(f"weights: {weights.detach()}")
        return total_loss, gradnorm_loss


class PCGradBalancer(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        self.num_losses = num_losses

    def forward(self, losses, shared_representation, is_excluded=None):
        losses = list(losses)
        if not torch.is_grad_enabled():
            return torch.tensor(sum(losses), device=losses[0].device)
        losses = list(losses)
        params = [p for p in shared_representation if p.requires_grad]

        if is_excluded is None:
            is_excluded = lambda p: False  # noqa

        shared_idx = []
        excluded_idx = []

        for i, p in enumerate(params):
            if is_excluded(p):
                excluded_idx.append(i)
            else:
                shared_idx.append(i)

        task_grads_shared = []
        task_grad_excluded = []
        valid_losses = []

        for loss in losses:
            if loss is None or not loss.requires_grad:
                continue

            grads = torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True
            )

            if all(g is None for g in grads):
                continue

            g_shared = []
            g_excluded = []

            for i, (g, p) in enumerate(zip(grads, params, strict=False)):
                if g is None:
                    g = torch.zeros_like(p)

                if i in shared_idx:
                    g_shared.append(g.reshape(-1))
                else:
                    g_excluded.append(g.reshape(-1))

            if len(g_shared) > 0:
                task_grads_shared.append(torch.cat(g_shared))
            if len(g_excluded) > 0:
                task_grad_excluded.append(torch.cat(g_excluded))

            valid_losses.append(loss)

        if len(valid_losses) == 0:
            return torch.tensor(0.0, device=losses[0].device)

        if len(task_grads_shared) > 0:
            task_grads_shared = torch.stack(task_grads_shared)

            if len(task_grad_excluded) == 1:
                final_shared = task_grads_shared[0]
            else:
                pc = task_grads_shared.clone()

            for i, _ in enumerate(pc):
                order = list(range(len(pc)))
                random.shuffle(order)

                for j in order:
                    if i == j:
                        continue

                dot = torch.dot(pc[i], task_grads_shared[j])
                if dot < 0:
                    coef = dot / (
                        torch.dot(task_grads_shared[j], task_grads_shared[j]) + 1e-8
                    )
                    pc[i] -= -coef * task_grads_shared[j]

            final_shared = pc.mean(dim=0)
        else:
            final_shared = None

        if len(task_grad_excluded) > 0:
            task_grad_excluded = torch.stack(task_grad_excluded)
            final_excluded = task_grad_excluded.mean(dim=0)
        else:
            final_excluded = None

        ptr_shared = 0
        ptr_excluded = 0
        for i, p in enumerate(params):
            numel = p.numel()

            if i in shared_idx:
                if final_shared is not None:
                    p.grad = final_shared[ptr_shared : ptr_shared + numel].view_as(p)
                    ptr_shared += numel
            else:
                if final_excluded is not None:
                    p.grad = final_excluded[
                        ptr_excluded : ptr_excluded + numel
                    ].view_as(p)
                    ptr_excluded += numel
        return sum(valid_losses).detach()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        loss_balancer: nn.Module,
        main_loss: str | None = None,
    ):
        super().__init__()
        self.loss_balancer = loss_balancer
        self.main_loss = main_loss

    def forward(
        self,
        losses: dict[str, torch.FloatTensor],
        dist: torch.FloatTensor,
        params=None,
        is_excluded=None,
        **kwargs,
    ):
        if isinstance(self.loss_balancer, WeightLossBalancer | UncertainlyLossBalancer):
            if self.main_loss:
                main_loss_value = losses.pop(self.main_loss)
                tasks_loss = self.loss_balancer(
                    losses.keys(),
                    losses.values(),
                )
                loss = main_loss_value + tasks_loss
            else:
                loss = self.loss_balancer(
                    losses.keys(),
                    losses.values(),
                )
        elif isinstance(self.loss_balancer, GradNormLossBalancer):
            if self.main_loss:
                main_loss_value = losses.pop(self.main_loss)
                total_loss, gradnorm_loss = self.loss_balancer(
                    losses.values(),
                    shared_representation=dist,
                )
                loss = main_loss_value + total_loss + gradnorm_loss
            else:
                total_loss, gradnorm_loss = self.loss_balancer(
                    losses.values(), shared_representation=dist
                )
                loss = total_loss + gradnorm_loss
        elif isinstance(self.loss_balancer, PCGradBalancer):
            loss = self.loss_balancer(
                losses.values(), shared_representation=params, is_excluded=is_excluded
            )
        return loss
