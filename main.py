import argparse
import csv
import random
import os
import time
from pathlib import Path
import numpy as np

import torch

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from data_loader import get_dataloaders
from hier_node_motif_gnnv3 import HierNodeMotifGNN
from losses import (
    compute_hier_multitask_classification_loss,
    compute_hier_regression_loss,
)


# ============================================================
# Task type config
# ============================================================

CLASSIFICATION_DATASETS = {
    "BACE",
    "BBBP",
    "ClinTox",
    "Tox21",
    "ToxCast",
    "SIDER",
    "MUV",
    "HIV",
}

REGRESSION_DATASETS = {
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
    "Lipo",
    "QM9",
}

DATASETS = (
    "FreeSolv",
    "ESOL",
    "Lipophilicity",
    "BACE",
    "BBBP",
    "SIDER",
    "Tox21",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
SPECIAL_SEEDS = {
    "SIDER": (5, 6, 7, 8, 9),
    "Tox21": (5, 6, 7, 8, 9),
}

# Final v3 fixed settings. The CLI only exposes dataset, learning rate, and seeds.
ROOT = "./dataset"
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 20
HIDDEN_DIM = 128
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5
LAMBDA_AUX = 1.0
MAX_POS_WEIGHT = 10.0
MAX_REGIONS = 32
SAVE_DIR = "./checkpoints"
NUM_WORKERS = 0

RAW_FIELDS = (
    "dataset",
    "max_regions",
    "seed",
    "task_type",
    "primary_metric",
    "best_epoch",
    "best_val_metric",
    "best_test_metric",
    "best_test_aux_metric",
    "train_size",
    "val_size",
    "test_size",
    "num_tasks",
    "hidden_dim",
    "batch_size",
    "lr",
    "weight_decay",
    "lambda_aux",
    "patience",
    "epochs",
    "region_cache_path",
    "checkpoint_path",
    "elapsed_seconds",
)

SUMMARY_FIELDS = (
    "dataset",
    "max_regions",
    "task_type",
    "seeds",
    "num_runs",
    "primary_metric",
    "val_mean",
    "val_std",
    "test_mean",
    "test_std",
    "test_aux_metric",
    "test_aux_mean",
    "test_aux_std",
)


def region_cache_dataset_name(dataset_name):
    if dataset_name == "Lipo":
        return "Lipophilicity"
    return dataset_name


def resolve_region_cache_path(dataset_name, root=ROOT, max_regions=MAX_REGIONS):
    candidate = (
        Path(root)
        / "molecular_regions"
        / f"{region_cache_dataset_name(dataset_name)}_regions_max{max_regions}.pt"
    )
    return str(candidate) if candidate.exists() else None


def infer_task_type(dataset_name: str):
    name = dataset_name.strip()

    if name in CLASSIFICATION_DATASETS:
        return "classification"

    if name in REGRESSION_DATASETS:
        return "regression"

    raise ValueError(
        f"Cannot infer task type for dataset '{dataset_name}'. "
        f"Please use --task_type classification or --task_type regression."
    )


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def report_std(values):
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def report_metric_name(task_type):
    return "roc_auc" if task_type == "classification" else "rmse"


def atomic_write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def load_raw_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_keys(rows):
    return {
        (row["dataset"], int(row["max_regions"]), int(row["seed"]))
        for row in rows
    }


def resolve_dataset_seeds(args, dataset_name):
    if args.seeds is not None:
        return tuple(args.seeds)
    return SPECIAL_SEEDS.get(dataset_name, DEFAULT_SEEDS)


def sort_result_rows(rows):
    dataset_rank = {dataset: index for index, dataset in enumerate(DATASETS)}
    return sorted(
        rows,
        key=lambda row: (
            dataset_rank.get(row["dataset"], len(DATASETS)),
            int(row["seed"]),
        ),
    )


def summarize_results(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], int(row["max_regions"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for dataset in DATASETS:
        group = grouped.get((dataset, MAX_REGIONS), [])
        if not group:
            continue

        group = sorted(group, key=lambda row: int(row["seed"]))
        val = np.array([float(row["best_val_metric"]) for row in group], dtype=np.float64)
        test = np.array([float(row["best_test_metric"]) for row in group], dtype=np.float64)
        aux_values = [
            float(row["best_test_aux_metric"])
            for row in group
            if row["best_test_aux_metric"] not in {"", "None", None}
        ]
        aux = np.array(aux_values, dtype=np.float64)
        task_type = group[0]["task_type"]

        summary.append(
            {
                "dataset": dataset,
                "max_regions": MAX_REGIONS,
                "task_type": task_type,
                "seeds": ",".join(str(int(row["seed"])) for row in group),
                "num_runs": len(group),
                "primary_metric": report_metric_name(task_type),
                "val_mean": float(val.mean()),
                "val_std": report_std(val),
                "test_mean": float(test.mean()),
                "test_std": report_std(test),
                "test_aux_metric": "mae" if aux.size else "",
                "test_aux_mean": "" if not aux.size else float(aux.mean()),
                "test_aux_std": "" if not aux.size else report_std(aux),
            }
        )
    return summary


def standardize_regression_labels(train_loader, val_loader, test_loader):
    """
    Standardize regression labels using train set statistics only.

    This modifies data.y in-place:
        y_std = (y - mean) / std

    Returns:
        y_mean, y_std
    """
    y_list = []

    for data in train_loader.dataset:
        y = data.y.view(1, -1).float()
        y_list.append(y)

    y_all = torch.cat(y_list, dim=0)

    y_mean = y_all.mean(dim=0)
    y_std = y_all.std(dim=0)
    y_std = torch.clamp(y_std, min=1e-6)

    for dataset in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
        for data in dataset:
            y = data.y.view(1, -1).float()
            data.y_raw = y.clone()
            data.y = (y - y_mean.view(1, -1)) / y_std.view(1, -1)

    return y_mean, y_std


def compute_multitask_pos_weight(train_loader, max_pos_weight=10.0):
    if max_pos_weight < 1.0:
        raise ValueError("max_pos_weight must be at least 1.0.")

    labels = torch.cat(
        [
            data.y.view(1, -1).float()
            for data in train_loader.dataset
        ],
        dim=0,
    )
    valid = ~torch.isnan(labels)
    positive = ((labels == 1) & valid).sum(dim=0).float()
    negative = ((labels == 0) & valid).sum(dim=0).float()

    pos_weight = torch.ones_like(positive)
    usable = (positive > 0) & (negative > 0)
    pos_weight[usable] = negative[usable] / positive[usable]
    return pos_weight.clamp(min=1.0, max=max_pos_weight)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()

    all_y = []
    all_prob = []

    for batch in loader:
        batch = batch.to(device)

        y = batch.y.view(batch.num_graphs, -1).float()

        out = model(batch)

        logits = out["final_logit"]
        prob = torch.sigmoid(logits)

        all_y.append(y.detach().cpu())
        all_prob.append(prob.detach().cpu())

    all_y = torch.cat(all_y, dim=0).numpy()
    all_prob = torch.cat(all_prob, dim=0).numpy()

    auc_list = []
    num_tasks = all_y.shape[1]

    for task_id in range(num_tasks):
        y_task = all_y[:, task_id]
        p_task = all_prob[:, task_id]

        valid_mask = ~np.isnan(y_task)

        y_task = y_task[valid_mask]
        p_task = p_task[valid_mask]

        if len(np.unique(y_task)) < 2:
            continue

        auc = roc_auc_score(y_task, p_task)
        auc_list.append(auc)

    mean_auc = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")

    return {"roc_auc": mean_auc}


@torch.no_grad()
def evaluate_regression(model, loader, device, y_mean, y_std):
    model.eval()

    all_y_raw = []
    all_pred_raw = []

    y_mean = y_mean.to(device).view(1, -1)
    y_std = y_std.to(device).view(1, -1)

    for batch in loader:
        batch = batch.to(device)

        y_std_target = batch.y.view(batch.num_graphs, -1).float()

        out = model(batch)

        pred_std = out["final_logit"]

        pred_raw = pred_std * y_std + y_mean
        y_raw = y_std_target * y_std + y_mean

        all_y_raw.append(y_raw.detach().cpu())
        all_pred_raw.append(pred_raw.detach().cpu())

    all_y_raw = torch.cat(all_y_raw, dim=0).numpy()
    all_pred_raw = torch.cat(all_pred_raw, dim=0).numpy()

    rmse_list = []
    mae_list = []

    num_tasks = all_y_raw.shape[1]

    for task_id in range(num_tasks):
        y_task = all_y_raw[:, task_id]
        p_task = all_pred_raw[:, task_id]

        valid_mask = ~np.isnan(y_task)

        y_task = y_task[valid_mask]
        p_task = p_task[valid_mask]

        if len(y_task) == 0:
            continue

        rmse = np.sqrt(mean_squared_error(y_task, p_task))
        mae = mean_absolute_error(y_task, p_task)

        rmse_list.append(rmse)
        mae_list.append(mae)

    mean_rmse = float(np.mean(rmse_list)) if len(rmse_list) > 0 else float("nan")
    mean_mae = float(np.mean(mae_list)) if len(mae_list) > 0 else float("nan")

    return {
        "rmse": mean_rmse,
        "mae": mean_mae,
    }


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_aux,
    task_type,
    pos_weight=None,
):
    model.train()

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    aux_loss_sum = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)

        y = batch.y.view(batch.num_graphs, -1).float()

        optimizer.zero_grad()

        out = model(batch)

        if task_type == "classification":
            loss_dict = compute_hier_multitask_classification_loss(
                out=out,
                y=y,
                lambda_aux=lambda_aux,
                pos_weight=pos_weight,
            )

        elif task_type == "regression":
            loss_dict = compute_hier_regression_loss(
                out=out,
                y=y,
                lambda_aux=lambda_aux,
            )

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        loss = loss_dict["loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        bs = batch.num_graphs

        total_loss_sum += loss.item() * bs
        main_loss_sum += loss_dict["main_loss"].item() * bs
        aux_loss_sum += loss_dict["aux_loss"].item() * bs
        n_graphs += bs

    logs = {
        "loss": total_loss_sum / n_graphs,
        "main_loss": main_loss_sum / n_graphs,
        "aux_loss": aux_loss_sum / n_graphs,
    }
    return logs


# ============================================================
# Single seed run
# ============================================================

def run_one_seed(args, seed):
    start_time = time.perf_counter()
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = infer_task_type(args.dataset)

    print("=" * 80)
    print(f"Start seed {seed}")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Task type: {task_type}")
    print(f"Device: {device}")
    print("Split: scaffold")
    print("use_hier: True")
    print(f"hidden_dim: {HIDDEN_DIM}")
    print("num_layers: 3")
    print("use_node_layers: (True, True, True)")
    print("use_motif_layers: (True, True, True)")
    print(f"max_regions: {MAX_REGIONS}")
    print("fusion_type: concat")
    print(f"lr: {args.lr}")
    print(f"lambda_aux: {LAMBDA_AUX}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    split_seed = seed
    region_cache_path = resolve_region_cache_path(args.dataset, root=args.root)
    print(
        "molecular_regions: "
        f"{region_cache_path if region_cache_path else 'not found; fallback to BRICS motifs'}"
    )

    train_loader, val_loader, test_loader, info = get_dataloaders(
        dataset_name=args.dataset,
        root=args.root,
        batch_size=BATCH_SIZE,
        split_type="scaffold",
        seed=split_seed,
        num_workers=NUM_WORKERS,
        remove_nan_labels=(task_type == "regression"),
        region_cache_path=region_cache_path,
    )

    if task_type == "regression":
        y_mean, y_std = standardize_regression_labels(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        print("Regression label standardization:")
        print(f"y_mean: {y_mean.tolist()}")
        print(f"y_std:  {y_std.tolist()}")

    else:
        y_mean, y_std = None, None

    if task_type == "classification":
        pos_weight = compute_multitask_pos_weight(
            train_loader,
            max_pos_weight=MAX_POS_WEIGHT,
        ).to(device)
        print(f"Training pos_weight: {pos_weight.cpu().tolist()}")
    else:
        pos_weight = None

    print("=" * 80)
    print("Dataset Info")
    print("=" * 80)
    for k, v in info.items():
        print(f"{k}: {v}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = HierNodeMotifGNN(
        node_dim=info["node_dim"],
        edge_dim=info["edge_dim"],
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_tasks=info["num_tasks"],
        task_type=task_type,
    ).to(device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    if task_type == "classification":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
        )
        better = lambda current, best: current > best
        best_metric = -1.0
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        better = lambda current, best: current < best
        best_metric = float("inf")

    # --------------------------------------------------------
    # Save path
    # --------------------------------------------------------
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(
        checkpoint_dir
        /
        (
            f"best_{args.dataset.lower()}_"
            f"{task_type}_"
            f"v3_seed{seed}_"
            f"split{split_seed}_"
            f"regions{MAX_REGIONS}_"
            "fusion_concat.pt"
        ),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    best_val_metric = best_metric
    best_epoch = 0
    bad_count = 0

    for epoch in range(1, EPOCHS + 1):
        train_logs = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_aux=LAMBDA_AUX,
            task_type=task_type,
            pos_weight=pos_weight,
        )

        if task_type == "classification":
            val_logs = evaluate_classification(
                model=model,
                loader=val_loader,
                device=device,
            )

            val_metric = val_logs["roc_auc"]

            scheduler.step(val_metric)

            metric_display = (
                f"val_auc={val_metric:.4f}"
            )

        else:
            val_logs = evaluate_regression(
                model=model,
                loader=val_loader,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
            )

            val_metric = val_logs["rmse"]

            scheduler.step(val_metric)

            metric_display = (
                f"val_rmse={val_logs['rmse']:.4f} | "
                f"val_mae={val_logs['mae']:.4f}"
            )

        if better(val_metric, best_val_metric):
            best_val_metric = val_metric
            best_epoch = epoch
            bad_count = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "model_class": "HierNodeMotifGNN",
                "model_config": {
                    "node_dim": info["node_dim"],
                    "edge_dim": info["edge_dim"],
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": 3,
                    "dropout": DROPOUT,
                    "num_tasks": info["num_tasks"],
                    "use_hier": True,
                    "use_node_layers": (True, True, True),
                    "use_motif_layers": (True, True, True),
                    "fusion_type": "concat",
                    "region_cache_path": region_cache_path,
                    "max_regions": MAX_REGIONS,
                    "task_type": task_type,
                },
                "seed": seed,
                "split_seed": split_seed,
                "task_type": task_type,
                "info": info,
                "best_epoch": best_epoch,
                "best_val_metric": best_val_metric,
            }
            if pos_weight is not None:
                checkpoint["pos_weight"] = pos_weight.cpu().tolist()

            if task_type == "regression":
                checkpoint["y_mean"] = y_mean.tolist()
                checkpoint["y_std"] = y_std.tolist()

            torch.save(checkpoint, save_path)

        else:
            bad_count += 1

        if task_type == "classification":
            best_display = (
                f"best_val_auc={best_val_metric:.4f}"
            )
        else:
            best_display = (
                f"best_val_rmse={best_val_metric:.4f}"
            )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Seed {seed} | "
                f"Epoch {epoch:03d} | "
                f"loss={train_logs['loss']:.4f} | "
                f"main={train_logs['main_loss']:.4f} | "
                f"aux={train_logs['aux_loss']:.4f} | "
                f"{metric_display} | "
                f"{best_display}"
            )

        if bad_count >= PATIENCE:
            print(f"Seed {seed} early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if task_type == "classification":
        test_logs = evaluate_classification(
            model=model,
            loader=test_loader,
            device=device,
        )
        best_test_metric = test_logs["roc_auc"]
        best_test_aux_metric = ""
    else:
        test_logs = evaluate_regression(
            model=model,
            loader=test_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
        )
        best_test_metric = test_logs["rmse"]
        best_test_aux_metric = test_logs["mae"]

    print("=" * 80)
    print(f"Seed {seed} Result")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")

    if task_type == "classification":
        print(f"Best val AUC:  {best_val_metric:.4f}")
        print(f"Best test AUC: {best_test_metric:.4f}")
    else:
        print(f"Best val RMSE:  {best_val_metric:.4f}")
        print(f"Best test RMSE: {best_test_metric:.4f}")
        print(f"Best test MAE:  {best_test_aux_metric:.4f}")

    print(f"Saved model: {save_path}")

    elapsed = time.perf_counter() - start_time

    return {
        "dataset": args.dataset,
        "max_regions": MAX_REGIONS,
        "seed": seed,
        "task_type": task_type,
        "primary_metric": report_metric_name(task_type),
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "best_test_metric": best_test_metric,
        "best_test_aux_metric": best_test_aux_metric,
        "train_size": info["train_size"],
        "val_size": info["val_size"],
        "test_size": info["test_size"],
        "num_tasks": info["num_tasks"],
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "lambda_aux": LAMBDA_AUX,
        "patience": PATIENCE,
        "epochs": EPOCHS,
        "region_cache_path": region_cache_path if region_cache_path else "",
        "checkpoint_path": save_path,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional single dataset to run. If omitted, runs all test datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="Datasets to run when --dataset is not set.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional manual seed override for every dataset. If omitted, "
            "SIDER/Tox21 use 5 6 7 8 9 and other datasets use 0 1 2 3 4."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--output_dir", default="./results/mainv3")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(args.datasets)
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [] if args.rerun else load_raw_rows(raw_path)
    done = set() if args.rerun else completed_keys(rows)

    print("=" * 80)
    print("Main v3 multi-dataset experiment")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"Fixed config: hidden_dim={HIDDEN_DIM}, batch_size={BATCH_SIZE}, epochs={EPOCHS}")
    print(f"lr: {args.lr}")
    print(f"Raw results: {raw_path}")
    print(f"Summary results: {summary_path}")

    for dataset_name in datasets:
        seeds = resolve_dataset_seeds(args, dataset_name)
        print("-" * 80)
        print(
            f"Dataset: {dataset_name} | task_type={infer_task_type(dataset_name)} | "
            f"seeds={list(seeds)}"
        )
        print("-" * 80)

        for seed in seeds:
            key = (dataset_name, MAX_REGIONS, seed)
            if key in done:
                print(f"Skip completed: dataset={dataset_name}, seed={seed}")
                continue

            run_args = argparse.Namespace(**vars(args))
            run_args.dataset = dataset_name

            row = run_one_seed(run_args, seed)
            rows.append(row)
            rows = sort_result_rows(rows)
            atomic_write_csv(raw_path, RAW_FIELDS, rows)
            atomic_write_csv(summary_path, SUMMARY_FIELDS, summarize_results(rows))
            done.add(key)

    rows = sort_result_rows(rows)
    atomic_write_csv(raw_path, RAW_FIELDS, rows)
    atomic_write_csv(summary_path, SUMMARY_FIELDS, summarize_results(rows))
    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary results to {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
