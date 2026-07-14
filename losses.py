import torch
import torch.nn.functional as F


# ============================================================
# Classification loss without KL
# ============================================================

def compute_hier_classification_loss(
    out,
    y,
    lambda_aux=0.2,
    pos_weight=None,
):
    """
    Binary classification loss with missing-label masking.

    total loss:
        masked BCE(final_logit, y)
        + lambda_aux * [
            masked BCE(layer1_logit, y)
            + masked BCE(layer2_logit, y)
            + masked BCE(layer3_logit, y)
        ]

    No KL term is used.
    """
    final_logit = out["final_logit"]

    if y.dim() == 1:
        y = y.view(-1, 1)
    y = y.float()

    main_loss = masked_bce_with_logits(
        final_logit,
        y,
        pos_weight=pos_weight,
    )

    aux_loss = (
        masked_bce_with_logits(
            out["layer1_logit"], y, pos_weight=pos_weight
        )
        + masked_bce_with_logits(
            out["layer2_logit"], y, pos_weight=pos_weight
        )
        + masked_bce_with_logits(
            out["layer3_logit"], y, pos_weight=pos_weight
        )
    )

    total_loss = main_loss + lambda_aux * aux_loss
    return {
        "loss": total_loss,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
    }


# ============================================================
# Masked BCE for multi-task datasets with missing labels
# ============================================================

def masked_bce_with_logits(logits, labels, pos_weight=None):
    """
    Task-balanced BCEWithLogits for multi-task labels with missing values.

    Each task is reduced independently, then valid task losses are averaged.
    This prevents tasks with more observed labels from dominating the loss.
    """
    labels = labels.float()
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
        logits = logits.view(-1, 1)

    if pos_weight is not None:
        pos_weight = pos_weight.to(
            device=logits.device,
            dtype=logits.dtype,
        ).view(-1)
        if pos_weight.numel() != labels.size(1):
            raise ValueError(
                "pos_weight must contain one value per task."
            )

    task_losses = []
    for task_idx in range(labels.size(1)):
        valid_mask = ~torch.isnan(labels[:, task_idx])
        if not valid_mask.any():
            continue

        task_logits = logits[valid_mask, task_idx]
        task_labels = labels[valid_mask, task_idx]
        task_pos_weight = (
            None
            if pos_weight is None
            else pos_weight[task_idx].view(1)
        )
        task_losses.append(
            F.binary_cross_entropy_with_logits(
                task_logits,
                task_labels,
                pos_weight=task_pos_weight,
            )
        )

    if not task_losses:
        return logits.sum() * 0.0

    return torch.stack(task_losses).mean()


def compute_hier_multitask_classification_loss(
    out,
    y,
    lambda_aux=0.2,
    pos_weight=None,
):
    """
    Multi-task classification loss for datasets like Tox21 / SIDER / ClinTox.
    No KL term is used.
    """
    final_logit = out["final_logit"]

    if y.dim() == 1:
        y = y.view(-1, 1)
    y = y.float()

    main_loss = masked_bce_with_logits(
        final_logit,
        y,
        pos_weight=pos_weight,
    )

    aux_loss = (
        masked_bce_with_logits(
            out["layer1_logit"], y, pos_weight=pos_weight
        )
        + masked_bce_with_logits(
            out["layer2_logit"], y, pos_weight=pos_weight
        )
        + masked_bce_with_logits(
            out["layer3_logit"], y, pos_weight=pos_weight
        )
    )

    total_loss = main_loss + lambda_aux * aux_loss
    return {
        "loss": total_loss,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
    }


# ============================================================
# Regression loss without KL
# ============================================================

def compute_hier_regression_loss(
    out,
    y,
    lambda_aux=0.2,
):
    """
    Regression loss.

    total loss:
        MSE(final_logit, y)
        + lambda_aux * [MSE(layer1_logit, y) + MSE(layer2_logit, y) + MSE(layer3_logit, y)]

    No KL term is used.
    """
    pred = out["final_logit"]

    if y.dim() == 1:
        y = y.view(-1, 1)
    y = y.float()

    main_loss = F.mse_loss(pred, y)

    aux_loss = (
        F.mse_loss(out["layer1_logit"], y)
        + F.mse_loss(out["layer2_logit"], y)
        + F.mse_loss(out["layer3_logit"], y)
    )

    total_loss = main_loss + lambda_aux * aux_loss
    return {
        "loss": total_loss,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
    }
