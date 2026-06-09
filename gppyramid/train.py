"""
Training utilities for GPPyramid Transformer.

These functions provide a compact reference implementation of masked loss,
single-fold training, prediction, and site-level cross-validation.
"""

import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import EcoDataset, compute_normalization_stats, normalize_samples
from .metrics import regression_metrics
from .model import GPPyramidTransformer


def masked_huber_loss(pred, target_norm, input_mask, delta=0.5):
    """Huber loss computed only over valid target and input positions."""
    valid = torch.isfinite(target_norm) & (~input_mask)
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[valid], target_norm[valid], delta=delta)


def build_model(config):
    """Construct the GPPyramid Transformer from a configuration dictionary."""
    return GPPyramidTransformer(
        hf_in=len(config["daily_vars"]),
        n_pft=len(config["pft_list"]),
        d_model=config.get("d_model", 64),
        num_heads=config.get("num_heads", 2),
        dropout=config.get("dropout", 0.3),
        target_len=config.get("target_len", 365),
        n_pyramid_layers=config.get("n_pyramid_layers", 2),
    )


def train_epoch(model, loader, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    for hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask, target_norm, _ in loader:
        hf, lai_s1, lai_s2, lai_s3, lai_s4 = hf.to(device), lai_s1.to(device), lai_s2.to(device), lai_s3.to(device), lai_s4.to(device)
        pft_id, input_mask, target_norm = pft_id.to(device), input_mask.to(device), target_norm.to(device)
        optimizer.zero_grad()
        pred = model(hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask)
        loss = masked_huber_loss(pred, target_norm, input_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, stats, device):
    """Evaluate a model and return R2, RMSE, MAE, and KGE."""
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    gpp_mean, gpp_std = stats["gpp_mean"], stats["gpp_std"]
    with torch.no_grad():
        for hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask, target_norm, _ in loader:
            hf, lai_s1, lai_s2, lai_s3, lai_s4 = hf.to(device), lai_s1.to(device), lai_s2.to(device), lai_s3.to(device), lai_s4.to(device)
            pft_id, input_mask, target_norm = pft_id.to(device), input_mask.to(device), target_norm.to(device)
            pred_norm = model(hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask)
            total_loss += masked_huber_loss(pred_norm, target_norm, input_mask).item()
            pred = pred_norm.cpu().numpy() * gpp_std + gpp_mean
            target = target_norm.cpu().numpy() * gpp_std + gpp_mean
            valid = (~input_mask.cpu().numpy()) & np.isfinite(target)
            preds.append(pred[valid])
            targets.append(target[valid])
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    metrics = regression_metrics(targets, preds)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train_single_fold(train_samples, test_samples, config, device):
    """Train one site-level cross-validation fold."""
    stats = compute_normalization_stats(train_samples)
    train_norm = normalize_samples(train_samples, stats, target_len=config.get("target_len", 365))
    test_norm = normalize_samples(test_samples, stats, target_len=config.get("target_len", 365))
    train_loader = DataLoader(EcoDataset(train_norm), batch_size=config.get("batch_size", 32), shuffle=True, drop_last=False)
    test_loader = DataLoader(EcoDataset(test_norm), batch_size=config.get("batch_size", 32), shuffle=False, drop_last=False)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4), weight_decay=config.get("weight_decay", 1e-4))
    best_state, best_loss = None, float("inf")
    patience_count = 0
    for epoch in range(config.get("epochs", 200)):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, test_loader, stats, device)
        print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | R2={val_metrics['R2']:.4f} | KGE={val_metrics['KGE']:.4f}")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= config.get("patience", 30):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, stats


def predict_samples(model, samples, stats, config, device):
    """Predict daily GPP for a list of site-year samples."""
    norm_samples = normalize_samples(samples, stats, target_len=config.get("target_len", 365))
    loader = DataLoader(EcoDataset(norm_samples), batch_size=1, shuffle=False)
    model.eval()
    results = []
    with torch.no_grad():
        for idx, (hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask, _, _) in enumerate(loader):
            hf, lai_s1, lai_s2, lai_s3, lai_s4 = hf.to(device), lai_s1.to(device), lai_s2.to(device), lai_s3.to(device), lai_s4.to(device)
            pft_id, input_mask = pft_id.to(device), input_mask.to(device)
            pred_norm = model(hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask)
            pred = pred_norm.cpu().numpy().ravel() * stats["gpp_std"] + stats["gpp_mean"]
            sample = samples[idx]
            results.append({"site": sample["meta"]["site"], "year": sample["meta"]["year"], "dates": sample["meta"]["dates"], "obs": sample["target"], "pred": pred})
    return results


def save_predictions(results, output_dir):
    """Save predictions as one CSV file per site."""
    os.makedirs(output_dir, exist_ok=True)
    site_records = {}
    for item in results:
        site = item["site"]
        site_records.setdefault(site, [])
        for date, obs, pred in zip(item["dates"], item["obs"], item["pred"]):
            site_records[site].append({"TIMESTAMP": date, "Obs_GPP": float(obs), "Pre_GPP": float(pred)})
    for site, records in site_records.items():
        df = pd.DataFrame(records).sort_values("TIMESTAMP")
        df.to_csv(os.path.join(output_dir, f"{site}.csv"), index=False)


def run_cross_validation(folds, config, device, output_dir=None):
    """Run site-level cross-validation and return pooled metrics."""
    all_results, fold_rows = [], []
    for fold_id in range(len(folds)):
        test_samples = folds[fold_id]
        train_samples = [sample for idx, fold in enumerate(folds) if idx != fold_id for sample in fold]
        print(f"Fold {fold_id + 1}/{len(folds)}")
        print(f"Training site-years: {len(train_samples)} | Test site-years: {len(test_samples)}")
        model, stats = train_single_fold(train_samples, test_samples, config, device)
        fold_results = predict_samples(model, test_samples, stats, config, device)
        y_true = np.concatenate([item["obs"] for item in fold_results])
        y_pred = np.concatenate([item["pred"] for item in fold_results])
        fold_metrics = regression_metrics(y_true, y_pred)
        fold_rows.append({"Fold": fold_id + 1, **fold_metrics})
        all_results.extend(fold_results)
    y_true_all = np.concatenate([item["obs"] for item in all_results])
    y_pred_all = np.concatenate([item["pred"] for item in all_results])
    global_metrics = regression_metrics(y_true_all, y_pred_all)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(fold_rows).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
        save_predictions(all_results, os.path.join(output_dir, "OBSvsPRE"))
    return global_metrics
