"""
Data interface for GPPyramid Transformer.

The functions in this file define a compact site-year data format and basic
normalization utilities. Users may adapt these functions to their own datasets.
"""

import copy
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import compute_scale_lengths


class EcoDataset(Dataset):
    """PyTorch dataset for site-year GPP sequences."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.FloatTensor(item["hf"]), torch.FloatTensor(item["lai_s1"]), torch.FloatTensor(item["lai_s2"]),
            torch.FloatTensor(item["lai_s3"]), torch.FloatTensor(item["lai_s4"]), torch.tensor(item["pft_id"], dtype=torch.long),
            torch.BoolTensor(item["input_mask"]), torch.FloatTensor(item["target_norm"]), item["meta"],
        )


def load_site_attributes(attr_file, pft_list):
    """Read site-level PFT information from a metadata table with SITE and PFT columns."""
    df = pd.read_csv(attr_file)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["SITE", "PFT"])
    pft2idx = {pft: idx for idx, pft in enumerate(pft_list)}
    site2pft = {}
    for _, row in df.iterrows():
        site = str(row["SITE"]).strip()
        pft = str(row["PFT"]).strip()
        if site and pft in pft2idx:
            site2pft[site] = pft2idx[pft]
    return site2pft


def _extract_lai_sequence(values, target_len):
    """Extract a fixed-length valid LAI sequence."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(valid) >= target_len:
        return valid[:target_len].astype(np.float32)
    pad = np.full(target_len - len(valid), valid[-1], dtype=np.float32)
    return np.concatenate([valid.astype(np.float32), pad])


def _interpolate_sequence(values, target_len):
    """Linearly interpolate a 1D sequence to the target length."""
    if len(values) == target_len:
        return values.astype(np.float32)
    x_src = np.linspace(0.0, 1.0, len(values))
    x_dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_dst, x_src, values).astype(np.float32)


def load_site_years(data_dir, attr_file, pft_list, daily_vars, lai_col="LAI4", target_col="GPP"):
    """Load per-site CSV files and convert them into site-year samples."""
    site2pft = load_site_attributes(attr_file, pft_list)
    samples = []
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    for file in csv_files:
        site = file.replace(".csv", "")
        if site not in site2pft:
            continue
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "TIMESTAMP" not in df.columns:
            continue
        required_cols = set(daily_vars + [target_col])
        if not required_cols.issubset(df.columns):
            continue
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["TIMESTAMP"])
        df = df[~((df["TIMESTAMP"].dt.month == 2) & (df["TIMESTAMP"].dt.day == 29))]
        df["YEAR"] = df["TIMESTAMP"].dt.year
        df = df.sort_values("TIMESTAMP")
        for year, group in df.groupby("YEAR"):
            group = group.head(365)
            if len(group) < 365:
                continue
            hf_raw = group[daily_vars].values.T.astype(np.float32)
            input_mask = np.any(~np.isfinite(hf_raw), axis=0)
            hf_clean = np.where(np.isfinite(hf_raw), hf_raw, 0.0)
            lai_raw = group[lai_col].values.astype(np.float32) if lai_col in group.columns else np.full(365, np.nan, dtype=np.float32)
            target = group[target_col].values.astype(np.float32)
            dates = group["TIMESTAMP"].dt.strftime("%Y%m%d").tolist()
            samples.append({
                "hf": hf_clean,
                "lai": lai_raw,
                "input_mask": input_mask,
                "target": target,
                "pft_id": site2pft[site],
                "meta": {"site": site, "year": int(year), "dates": dates},
            })
    return samples


def compute_normalization_stats(samples):
    """Estimate normalization statistics from training samples only."""
    hf_values, gpp_values, lai_values = [], [], []
    for sample in samples:
        valid_days = ~sample["input_mask"]
        hf_values.append(sample["hf"][:, valid_days])
        gpp = sample["target"]
        gpp_values.append(gpp[np.isfinite(gpp)])
        lai = sample["lai"]
        lai_values.append(lai[np.isfinite(lai)])
    hf_stack = np.concatenate(hf_values, axis=1)
    gpp_stack = np.concatenate(gpp_values)
    lai_stack = np.concatenate(lai_values)
    return {
        "hf_mean": hf_stack.mean(axis=1, keepdims=True),
        "hf_std": hf_stack.std(axis=1, keepdims=True) + 1e-6,
        "gpp_mean": float(gpp_stack.mean()),
        "gpp_std": float(gpp_stack.std() + 1e-6),
        "lai_mean": float(lai_stack.mean()) if lai_stack.size > 0 else 0.0,
        "lai_std": float(lai_stack.std() + 1e-6) if lai_stack.size > 0 else 1.0,
    }


def normalize_samples(samples, stats, target_len=365):
    """Normalize drivers, LAI, and GPP target sequences."""
    scale_lengths = compute_scale_lengths(target_len)
    output = []
    for sample in samples:
        item = copy.deepcopy(sample)
        item["hf"] = (item["hf"] - stats["hf_mean"]) / stats["hf_std"]
        item["target_norm"] = (item["target"] - stats["gpp_mean"]) / stats["gpp_std"]
        lai_native = _extract_lai_sequence(item["lai"], scale_lengths[2])
        lai_native = (lai_native - stats["lai_mean"]) / stats["lai_std"]
        item["lai_s1"] = _interpolate_sequence(lai_native, scale_lengths[0]).reshape(1, scale_lengths[0])
        item["lai_s2"] = _interpolate_sequence(lai_native, scale_lengths[1]).reshape(1, scale_lengths[1])
        item["lai_s3"] = lai_native.reshape(1, scale_lengths[2])
        item["lai_s4"] = _interpolate_sequence(lai_native, scale_lengths[3]).reshape(1, scale_lengths[3])
        output.append(item)
    return output


def build_site_folds(samples, n_folds=5):
    """Build deterministic site-level folds with approximate PFT balance."""
    site_info = {}
    for sample in samples:
        site = sample["meta"]["site"]
        if site not in site_info:
            site_info[site] = {"pft_id": sample["pft_id"], "count": 0}
        site_info[site]["count"] += 1
    pft_groups = {}
    for site, info in site_info.items():
        pft_groups.setdefault(info["pft_id"], []).append(site)
    fold_loads = np.zeros(n_folds)
    site2fold = {}
    for _, sites in sorted(pft_groups.items()):
        sites = sorted(sites, key=lambda s: site_info[s]["count"], reverse=True)
        for site in sites:
            fold_id = int(np.argmin(fold_loads))
            site2fold[site] = fold_id
            fold_loads[fold_id] += site_info[site]["count"]
    folds = [[] for _ in range(n_folds)]
    for sample in samples:
        site = sample["meta"]["site"]
        folds[site2fold[site]].append(sample)
    return folds, site2fold
