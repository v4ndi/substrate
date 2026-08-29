from typing import Literal

import torch
import torch.nn as nn


class KLDLoss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.t_loss = nn.CrossEntropyLoss()
        self.c_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        dist: torch.FloatTensor,
        targets: torch.LongTensor,
        **kwargs,
    ):
        mu_c, var_c = (
            dist[is_treat == 0].mean(0),
            dist[is_treat == 0].var(0, unbiased=False),
        )
        mu_t, var_t = (
            dist[is_treat == 1].mean(0),
            dist[is_treat == 1].var(0, unbiased=False),
        )

        var_c, var_t = (
            torch.clamp(var_c, min=1e-6).detach(),
            torch.clamp(var_t, min=1e-6).detach(),
        )

        kld_loss = 0.5 * torch.sum(
            torch.log(var_t / var_c) + (var_c + (mu_c - mu_t) ** 2) / var_t - 1
        )

        if dist[is_treat == 0].shape[0] < 2 or dist[is_treat == 1].shape[0] < 2:
            kld_loss = torch.tensor(0.0, device=dist.device)

        total_loss = (
            self.c_loss(logits[is_treat == 0], targets[is_treat == 0])
            + self.t_loss(logits[is_treat == 1], targets[is_treat == 1])
            + self.alpha * kld_loss
        )
        return total_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha: float, separate_heads: bool = False):
        super().__init__()
        self.alpha = alpha
        self.separate_heads = separate_heads
        if separate_heads:
            self.t_loss = nn.CrossEntropyLoss()
            self.c_loss = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor,
        **kwargs,
    ):
        pos = logits[(is_treat.bool())][:, 1]
        neg = logits[(~is_treat.bool())][:, 0]
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        contrastive_loss = torch.mean(-torch.log(torch.sigmoid(diff)))

        if (
            logits[(is_treat.bool())].shape[0] < 2
            or logits[(~is_treat.bool())].shape[0] < 2
        ):
            contrastive_loss = torch.tensor(0.0, device=logits.device)

        if self.separate_heads:
            ce_loss = self.c_loss(logits[is_treat == 0], targets[is_treat == 0])
            +self.t_loss(logits[is_treat == 1], targets[is_treat == 1])
        else:
            ce_loss = self.ce(logits, targets)
        total_loss = ce_loss + self.alpha * contrastive_loss

        return total_loss


class GradNormLossBalancer(nn.Module):
    def __init__(self, num_losses, alpha=1.5, renormilize_weights=False):
        super().__init__()
        self.alpha = alpha
        # self.weights = nn.Parameter(torch.ones(num_losses))
        self.log_weights = nn.Parameter(torch.zeros(num_losses))
        self.num_losses = num_losses
        self.initial_losses = None
        self.renormilize_weights = renormilize_weights

    def forward(self, losses, shared_representation):
        if self.initial_losses is None:
            self.initial_losses = torch.tensor(
                [loss.detach().item() for loss in losses], device=losses[0].device
            )
        weights = torch.exp(self.log_weights)
        if self.renormilize_weights:
            weights = weights * (self.num_losses / weights.sum())
        weighted_loss = [w * loss for w, loss in zip(weights, losses, strict=False)]
        total_loss = sum(weighted_loss)

        grad_norms = []
        for wl in weighted_loss:
            if wl.detach().item() == 0:
                grad_norms.append(torch.tensor(0.0, device=wl.device))
            else:
                g = torch.autograd.grad(
                    wl,
                    shared_representation,
                    retain_graph=True,
                    create_graph=True,  # allow_unused=True
                )[0]
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


class KLDxContrastiveLoss(nn.Module):
    def __init__(self, loss_balancer: GradNormLossBalancer, ce_balance=False):
        super().__init__()
        self.loss_balancer = loss_balancer
        self.t_loss = nn.CrossEntropyLoss()
        self.c_loss = nn.CrossEntropyLoss()
        self.ce_balance = ce_balance

    def _kld_forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        dist: torch.FloatTensor,
        targets: torch.LongTensor,
    ):
        mu_c, var_c = (
            dist[is_treat == 0].mean(0),
            dist[is_treat == 0].var(0, unbiased=False),
        )
        mu_t, var_t = (
            dist[is_treat == 1].mean(0),
            dist[is_treat == 1].var(0, unbiased=False),
        )

        var_c, var_t = (
            torch.clamp(var_c, min=1e-6).detach(),
            torch.clamp(var_t, min=1e-6).detach(),
        )

        kld_loss = 0.5 * torch.sum(
            torch.log(var_t / var_c) + (var_c + (mu_c - mu_t) ** 2) / var_t - 1
        )
        return kld_loss

    def _contrastive_forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor,
    ):
        pos = logits[(is_treat.bool())][:, 1]
        neg = logits[(~is_treat.bool())][:, 0]
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        contrastive_loss = torch.mean(-torch.log(torch.sigmoid(diff)))
        return contrastive_loss

    def forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        dist: torch.FloatTensor,
        targets: torch.LongTensor,
        **kwargs,
    ):
        control_loss = self.c_loss(logits[~is_treat.bool()], targets[~is_treat.bool()])
        treat_loss = self.t_loss(logits[is_treat.bool()], targets[is_treat.bool()])
        kld_loss = self._kld_forward(
            logits=logits, is_treat=is_treat, dist=dist, targets=targets
        )
        contrastive_loss = self._contrastive_forward(
            logits=logits, is_treat=is_treat, targets=targets
        )

        if dist[is_treat == 0].shape[0] < 2 or dist[is_treat == 1].shape[0] < 2:
            control_loss = torch.tensor(0.0, device=dist.device)
            treat_loss = torch.tensor(0.0, device=dist.device)
            kld_loss = torch.tensor(0.0, device=dist.device)
            contrastive_loss = torch.tensor(0.0, device=dist.device)

        if self.ce_balance:
            total_loss, gradnorm_loss = self.loss_balancer(
                [control_loss, treat_loss, kld_loss, contrastive_loss],
                shared_representation=dist,
            )
            loss = total_loss + gradnorm_loss
        else:
            total_aux_loss, gradnorm_loss = self.loss_balancer(
                [kld_loss, contrastive_loss], shared_representation=dist
            )
            loss = control_loss + treat_loss + total_aux_loss + gradnorm_loss
        print(loss.detach())
        return loss


class KLDxContrastiveGridLoss(nn.Module):
    def __init__(self, alpha1: float, alpha2: float):
        super().__init__()
        self.t_loss = nn.CrossEntropyLoss()
        self.c_loss = nn.CrossEntropyLoss()

        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def _kld_forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        dist: torch.FloatTensor,
        targets: torch.LongTensor,
    ):
        mu_c, var_c = (
            dist[is_treat == 0].mean(0),
            dist[is_treat == 0].var(0, unbiased=False),
        )
        mu_t, var_t = (
            dist[is_treat == 1].mean(0),
            dist[is_treat == 1].var(0, unbiased=False),
        )

        var_c, var_t = (
            torch.clamp(var_c, min=1e-6).detach(),
            torch.clamp(var_t, min=1e-6).detach(),
        )

        kld_loss = 0.5 * torch.sum(
            torch.log(var_t / var_c) + (var_c + (mu_c - mu_t) ** 2) / var_t - 1
        )
        return kld_loss

    def _contrastive_forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        targets: torch.LongTensor,
    ):
        pos = logits[(is_treat.bool())][:, 1]
        neg = logits[(~is_treat.bool())][:, 0]
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        contrastive_loss = torch.mean(-torch.log(torch.sigmoid(diff)))
        return contrastive_loss

    def forward(
        self,
        logits: torch.FloatTensor,
        is_treat: torch.LongTensor,
        dist: torch.FloatTensor,
        targets: torch.LongTensor,
        **kwargs,
    ):
        control_loss = self.c_loss(logits[~is_treat.bool()], targets[~is_treat.bool()])
        treat_loss = self.t_loss(logits[is_treat.bool()], targets[is_treat.bool()])
        kld_loss = self._kld_forward(
            logits=logits, is_treat=is_treat, dist=dist, targets=targets
        )
        contrastive_loss = self._contrastive_forward(
            logits=logits, is_treat=is_treat, targets=targets
        )

        if dist[is_treat == 0].shape[0] < 2 or dist[is_treat == 1].shape[0] < 2:
            control_loss = torch.tensor(0.0, device=dist.device)
            treat_loss = torch.tensor(0.0, device=dist.device)
            kld_loss = torch.tensor(0.0, device=dist.device)
            contrastive_loss = torch.tensor(0.0, device=dist.device)

        total_loss = (
            control_loss
            + treat_loss
            + self.alpha1 * kld_loss
            + self.alpha2 * contrastive_loss
        )
        print(total_loss)
        return total_loss


class ResearchLosses(nn.Module):
    def __init__(
        self,
        loss: Literal[
            "KLDLoss",
            "ContrastiveLoss",
            "KLDxContrastiveLoss",
            "KLDxContrastiveGridLoss",
        ],
    ):
        super().__init__()
        self.loss = loss

    def forward(self, **kwargs):
        loss = self.loss(**kwargs)
        return loss
