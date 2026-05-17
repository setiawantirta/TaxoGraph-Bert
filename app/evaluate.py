"""
evaluate.py
-----------
Modul evaluasi dan analisis untuk TaxoGraphBERT.

Fungsi utama:
  - evaluate_on_mockrobiota()     : benchmark komunitas mikrobial sintetik
  - evaluate_ood_holdout()        : evaluasi OOD detection + AUROC/DeLong CI
  - run_ablation_study()          : ablasi arsitektur
  - plot_zipfian_distribution()   : distribusi Zipf sebelum/sesudah rollup
  - plot_ood_distributions()      : distribusi skor OOD in-distribution vs OOD
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MOCKROBIOTA BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_mockrobiota(
    model,
    tokenizer,
    label_enc,
    mock_dir: str | Path,
    device: str | torch.device = "cpu",
    mock_ids: Optional[List[int]] = None,
    conf_thresholds: Optional[Dict[str, float]] = None,
    max_reads_per_mock: int = 2000,
) -> "pd.DataFrame":
    """
    Evaluasi TaxoGraphBERT pada seluruh dataset Mockrobiota.

    Untuk setiap dataset mock:
      1. Tokenisasi sekuens → forward pass → `predict_with_abstention()`
      2. Bandingkan prediksi dengan expected-taxonomy.tsv (ground truth)
      3. Hitung precision/recall/F1 per rank + akurasi keseluruhan

    Parameter:
        model           : TaxoGraphBERT (mode eval)
        tokenizer       : DualTokenizer atau KmerTokenizer
        label_enc       : HierarchicalLabelEncoder (untuk decode label)
        mock_dir        : direktori root mockrobiota (berisi mock-1/, mock-2/, ...)
        device          : device untuk inferensi
        mock_ids        : subset ID yang dievaluasi (default: semua yang tersedia)
        conf_thresholds : threshold per rank untuk abstention

    Return:
        DataFrame dengan kolom: mock_id, rank, precision, recall, f1, n_samples, abstain_rate
    """
    import pandas as pd
    from data_acquisition import load_mockrobiota_dataset, MOCKROBIOTA_16S_IDS
    from sklearn.metrics import precision_recall_fscore_support

    if mock_ids is None:
        # Auto-detect mock IDs dari folder yang ada
        mock_dir = Path(mock_dir)
        mock_ids = [
            mid for mid in MOCKROBIOTA_16S_IDS
            if (mock_dir / f"mock-{mid}").exists()
        ]

    device = torch.device(device)
    model.eval()
    _raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    RANK_NAMES = label_enc.ranks  # ["Phylum", "Class", ..., "Species"]

    all_rows = []

    for mid in mock_ids:
        try:
            data = load_mockrobiota_dataset(mock_dir, mid, max_reads=max_reads_per_mock)
        except FileNotFoundError as exc:
            logger.warning(f"[mockrobiota] mock-{mid} tidak ditemukan: {exc}")
            continue

        sequences = data["sequences"]
        taxonomy  = data["taxonomy"]  # list of dicts per OTU

        if not sequences:
            logger.warning(f"[mockrobiota] mock-{mid}: tidak ada sekuens, skip.")
            continue

        # ── Tokenisasi & prediksi ────────────────────────────────────────────
        all_preds     = []   # list of dicts {rank: label}
        abstain_count = 0

        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i: i + batch_size]
            enc = tokenizer.batch_encode(batch_seqs)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                results = _raw_model.predict_with_abstention(
                    input_ids,
                    attention_mask,
                    conf_thresholds=conf_thresholds,
                    label_encoder=label_enc,
                )

            for r in results:
                all_preds.append(r["predictions"])
                if r["abstain_rank"] is not None:
                    abstain_count += 1

            del input_ids, attention_mask
            if device.type == "cuda":
                torch.cuda.empty_cache()

        n_samples    = len(sequences)
        abstain_rate = abstain_count / max(n_samples, 1)

        # ── Susun ground truth dari expected-taxonomy.tsv ───────────────────
        # Ground truth per-OTU; repeat berdasarkan read count jika ada
        # Untuk simplicity: assume 1 entry = 1 read jika jumlah = n_samples
        # Jika tidak cocok, gunakan modus per rank sebagai referensi
        gt_by_rank: Dict[str, List[str]] = {r: [] for r in RANK_NAMES}

        if len(taxonomy) == n_samples:
            # 1:1 mapping
            for entry in taxonomy:
                for rank in RANK_NAMES:
                    gt_by_rank[rank].append(str(entry.get(rank, "")).strip())
        else:
            # Gunakan entry pertama (komunitas ideal) sebagai referensi tunggal
            for rank in RANK_NAMES:
                ref_label = taxonomy[0].get(rank, "") if taxonomy else ""
                gt_by_rank[rank] = [str(ref_label).strip()] * n_samples

        pred_by_rank: Dict[str, List[str]] = {r: [] for r in RANK_NAMES}
        for pred_dict in all_preds:
            for rank in RANK_NAMES:
                pred_by_rank[rank].append(pred_dict.get(rank, ""))

        # ── Hitung metrik per rank ───────────────────────────────────────────
        for rank in RANK_NAMES:
            y_true = gt_by_rank[rank]
            y_pred = pred_by_rank[rank]

            # Hanya hitung metrik pada sampel yang punya ground truth
            valid = [(t, p) for t, p in zip(y_true, y_pred) if t]
            if not valid:
                continue
            y_true_v, y_pred_v = zip(*valid)

            try:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_v, y_pred_v, average="macro", zero_division=0
                )
            except Exception:
                prec = rec = f1 = 0.0

            all_rows.append({
                "mock_id":      mid,
                "rank":         rank,
                "precision":    prec,
                "recall":       rec,
                "f1":           f1,
                "n_samples":    len(valid),
                "abstain_rate": abstain_rate,
            })

        logger.info(
            f"[mockrobiota] mock-{mid}: {n_samples} reads, "
            f"abstain={abstain_rate:.2%}"
        )

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# OOD TEMPORAL HOLD-OUT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ood_holdout(
    model,
    tokenizer,
    label_enc,
    holdout_fasta: str | Path,
    val_dl: DataLoader,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Evaluasi kemampuan OOD detection menggunakan temporal hold-out NCBI.

    Menghitung:
      - AUROC (OOD skor in-dist vs OOD)
      - DeLong 95% CI untuk AUROC
      - FPR@TPR95 (False Positive Rate saat TPR=95%)
      - AUPRC (Area Under Precision-Recall Curve)

    Parameter:
        model          : TaxoGraphBERT (mode eval)
        tokenizer      : DualTokenizer atau KmerTokenizer
        label_enc      : HierarchicalLabelEncoder
        holdout_fasta  : path ke file FASTA OOD (dari fetch_ncbi_temporal_holdout)
        val_dl         : DataLoader dari set validasi (in-distribution)
        device         : device untuk inferensi

    Return:
        dict dengan key: auroc, auroc_ci_lo, auroc_ci_hi, fpr_at_tpr95, auprc,
                         n_indist, n_ood
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    device = torch.device(device)
    model.eval()

    # ── Kumpulkan skor OOD dari validation set (in-distribution) ───────────
    indist_scores = _collect_ood_scores(model, val_dl, device)

    # ── Kumpulkan skor OOD dari holdout FASTA (out-of-distribution) ────────
    ood_scores = _collect_ood_scores_from_fasta(model, tokenizer, holdout_fasta, device)

    if len(ood_scores) == 0:
        logger.warning("[ood_eval] File FASTA OOD kosong, tidak ada evaluasi.")
        return {}

    n_indist = len(indist_scores)
    n_ood    = len(ood_scores)

    # Label: 0 = in-distribution, 1 = OOD
    y_true  = np.array([0] * n_indist + [1] * n_ood, dtype=np.int32)
    y_score = np.concatenate([indist_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    # DeLong CI (bootstrap karena implementasi DeLong penuh membutuhkan statsmodels)
    ci_lo, ci_hi = _delong_ci_bootstrap(y_true, y_score, n_boot=1000, alpha=0.05)

    # FPR@TPR95
    fpr_at_95 = _compute_fpr_at_tpr(y_true, y_score, tpr_target=0.95)

    result = {
        "auroc":          float(auroc),
        "auroc_ci_lo":    float(ci_lo),
        "auroc_ci_hi":    float(ci_hi),
        "fpr_at_tpr95":   float(fpr_at_95),
        "auprc":          float(auprc),
        "n_indist":       n_indist,
        "n_ood":          n_ood,
    }

    logger.info(
        f"[ood_eval] AUROC={auroc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] "
        f"FPR@TPR95={fpr_at_95:.4f} AUPRC={auprc:.4f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    cfg,
    train_dl: DataLoader,
    val_dl: DataLoader,
    label_enc,
    edge_index: Tensor,
    leaf_node_ids: Optional[Tensor] = None,
    variants: Optional[List[dict]] = None,
    n_epochs: int = 5,
) -> "pd.DataFrame":
    """
    Jalankan ablation study untuk membandingkan varian arsitektur.

    Setiap varian dilatih untuk n_epochs epoch; validasi pada val_dl.
    Metrik utama: F1-macro, F1-Species.

    Parameter:
        cfg          : Config object
        train_dl     : DataLoader training
        val_dl       : DataLoader validasi
        label_enc    : HierarchicalLabelEncoder
        edge_index   : taxonomy graph edges
        leaf_node_ids: Species-level leaf node IDs
        variants     : list of dict, setiap dict berisi override config model
                       Contoh: [{"name": "no_hgnn", "use_hgnn": False}, ...]
                       Default: suite ablasi standar dari paper
        n_epochs     : jumlah epoch per varian (5 cukup untuk perbandingan)

    Return:
        DataFrame dengan kolom: variant, f1_macro, f1_species, f1_genus,
                                f1_phylum, best_epoch
    """
    import copy
    import pandas as pd
    from taxograph_bert import build_model
    from trainer import Trainer

    if variants is None:
        # Suite ablasi standar (sesuai Section 4 paper)
        variants = [
            {"name": "full_model",     "use_lora": True,  "use_hgnn": True,  "use_hyperbolic": True},
            {"name": "no_lora",        "use_lora": False, "use_hgnn": True,  "use_hyperbolic": True},
            {"name": "no_hgnn",        "use_lora": True,  "use_hgnn": False, "use_hyperbolic": True},
            {"name": "no_hyperbolic",  "use_lora": True,  "use_hgnn": True,  "use_hyperbolic": False},
            {"name": "no_lora_no_hgnn","use_lora": False, "use_hgnn": False, "use_hyperbolic": True},
        ]

    rows = []

    for variant in variants:
        name = variant.get("name", "unknown")
        logger.info(f"[ablation] Menjalankan varian: {name}")

        # Buat config sementara dengan override
        cfg_copy = copy.deepcopy(cfg)
        for key, val in variant.items():
            if key == "name":
                continue
            if hasattr(cfg_copy.model, key):
                setattr(cfg_copy.model, key, val)

        # Build model
        try:
            model = build_model(cfg_copy, label_enc, edge_index, leaf_node_ids)
        except Exception as exc:
            logger.error(f"[ablation] Gagal build model varian {name}: {exc}")
            continue

        # Train singkat
        trainer = Trainer(model, train_dl, val_dl, cfg_copy)
        trainer.setup_optimizers()

        best_f1_macro   = 0.0
        best_f1_species = 0.0
        best_f1_genus   = 0.0
        best_f1_phylum  = 0.0
        best_epoch      = 0

        for epoch in range(1, n_epochs + 1):
            trainer._train_one_epoch(epoch)
            val_metrics = trainer._validate(epoch)

            macro = val_metrics.get("f1_macro", 0.0)
            if macro > best_f1_macro:
                best_f1_macro   = macro
                best_f1_species = val_metrics.get("f1_Species", 0.0)
                best_f1_genus   = val_metrics.get("f1_Genus", 0.0)
                best_f1_phylum  = val_metrics.get("f1_Phylum", 0.0)
                best_epoch      = epoch

        rows.append({
            "variant":     name,
            "f1_macro":    best_f1_macro,
            "f1_species":  best_f1_species,
            "f1_genus":    best_f1_genus,
            "f1_phylum":   best_f1_phylum,
            "best_epoch":  best_epoch,
        })
        logger.info(f"[ablation] {name}: F1-macro={best_f1_macro:.4f}")

    df = pd.DataFrame(rows)
    logger.info(f"[ablation] Selesai. {len(rows)} varian dievaluasi.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────

def plot_zipfian_distribution(
    before_df,
    after_df,
    save_path: Optional[str | Path] = None,
    rank: str = "Species",
) -> None:
    """
    Plot distribusi Zipfian label taksonomi sebelum dan sesudah taxonomic rollup.

    Sumbu X: rank frekuensi (log scale)
    Sumbu Y: jumlah sampel (log scale)

    Parameter:
        before_df : DataFrame sebelum rollup (harus memiliki kolom `rank`)
        after_df  : DataFrame sesudah rollup
        save_path : path file output (PNG/SVG). Jika None, tampilkan interaktif.
        rank      : kolom rank yang divisualisasikan (default: "Species")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib diperlukan: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, title in [
        (axes[0], before_df, f"Sebelum Rollup ({rank})"),
        (axes[1], after_df,  f"Sesudah Rollup ({rank})"),
    ]:
        if rank not in df.columns:
            ax.set_title(f"{title}\n(kolom '{rank}' tidak ditemukan)")
            continue

        counts = df[rank].value_counts().values
        counts_sorted = np.sort(counts)[::-1]
        ranks_x = np.arange(1, len(counts_sorted) + 1)

        ax.loglog(ranks_x, counts_sorted, "o-", markersize=2, alpha=0.6)
        ax.set_xlabel("Rank frekuensi (log)")
        ax.set_ylabel("Jumlah sampel (log)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.95, 0.95,
            f"N labels = {len(counts_sorted)}\nMin = {counts_sorted[-1]}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Zipf plot disimpan: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_ood_distributions(
    indist_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: Optional[float] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plot distribusi skor OOD untuk in-distribution vs OOD samples.

    Menampilkan histogram/KDE kedua distribusi beserta threshold delta.
    Berguna untuk visualisasi efektivitas OOD detection.

    Parameter:
        indist_scores : array skor OOD untuk sampel in-distribution
        ood_scores    : array skor OOD untuk sampel OOD
        threshold     : threshold delta yang digunakan model (garis vertikal)
        save_path     : path output. Jika None, tampilkan interaktif.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib diperlukan: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        indist_scores, bins=50, alpha=0.6,
        color="steelblue", label=f"In-distribution (n={len(indist_scores)})",
        density=True,
    )
    ax.hist(
        ood_scores, bins=50, alpha=0.6,
        color="tomato", label=f"OOD (n={len(ood_scores)})",
        density=True,
    )

    if threshold is not None:
        ax.axvline(
            x=threshold, color="black", linestyle="--", linewidth=1.5,
            label=f"Threshold δ = {threshold:.3f}"
        )

    ax.set_xlabel("Skor OOD (jarak hiperbolik ke leaf terdekat)")
    ax.set_ylabel("Densitas")
    ax.set_title("Distribusi Skor OOD: In-Distribution vs OOD")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"OOD distribution plot disimpan: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER PRIVAT
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_ood_scores(model, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Kumpulkan skor OOD dari seluruh DataLoader (in-distribution)."""
    scores_list = []
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids, attention_mask, update_graph=False)
        scores_list.append(out["ood_score"].cpu().numpy())
        del input_ids, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(scores_list) if scores_list else np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER COMPARISON (pretrain vs scratch)
# ─────────────────────────────────────────────────────────────────────────────

def plot_encoder_comparison(
    pretrain_csv: "str | Path",
    scratch_csv: "str | Path",
    save_path: Optional["str | Path"] = None,
) -> None:
    """
    Plot perbandingan pretrain vs scratch encoder dari dua CSV log training.

    Menampilkan 2 subplot:
      - Kiri  : val F1-macro per epoch (pretrain vs scratch)
      - Kanan : train loss_total per epoch (pretrain vs scratch)

    Parameter:
        pretrain_csv : path CSV log training mode 'pretrain'
        scratch_csv  : path CSV log training mode 'scratch'
        save_path    : path output PNG. Jika None, tampilkan interaktif.
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("pandas dan matplotlib diperlukan: pip install pandas matplotlib")
        return

    from pathlib import Path as _Path

    df_pre = pd.read_csv(_Path(pretrain_csv))
    df_scr = pd.read_csv(_Path(scratch_csv))

    val_pre   = df_pre[df_pre["phase"] == "val"].reset_index(drop=True)
    val_scr   = df_scr[df_scr["phase"] == "val"].reset_index(drop=True)
    train_pre = df_pre[df_pre["phase"] == "train"].reset_index(drop=True)
    train_scr = df_scr[df_scr["phase"] == "train"].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Val F1-macro ──────────────────────────────────────────────────────
    axes[0].plot(val_pre["epoch"], val_pre["f1_macro"],
                 color="steelblue", linewidth=2, label="Pretrain (DNABERT-2)")
    axes[0].plot(val_scr["epoch"], val_scr["f1_macro"],
                 color="tomato", linewidth=2, linestyle="--", label="Scratch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val F1-macro")
    axes[0].set_title("Validation F1-macro: Pretrain vs Scratch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Train Loss ────────────────────────────────────────────────────────
    axes[1].plot(train_pre["epoch"], train_pre["loss_total"],
                 color="steelblue", linewidth=2, label="Pretrain (DNABERT-2)")
    axes[1].plot(train_scr["epoch"], train_scr["loss_total"],
                 color="tomato", linewidth=2, linestyle="--", label="Scratch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Loss")
    axes[1].set_title("Training Loss: Pretrain vs Scratch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Anotasi best val F1 per mode
    for df_val, color, label in [
        (val_pre, "steelblue", "Pretrain"),
        (val_scr, "tomato", "Scratch"),
    ]:
        if len(df_val):
            best_idx = df_val["f1_macro"].idxmax()
            best_ep  = df_val.loc[best_idx, "epoch"]
            best_f1  = df_val.loc[best_idx, "f1_macro"]
            axes[0].annotate(
                f"{label} best\nF1={best_f1:.4f} @ ep{int(best_ep)}",
                xy=(best_ep, best_f1),
                xytext=(best_ep + 1, best_f1 - 0.02),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )

    plt.suptitle("Encoder Comparison: Pretrain (DNABERT-2) vs Scratch", fontsize=13)
    plt.tight_layout()

    if save_path:
        from pathlib import Path as _Path2
        plt.savefig(_Path2(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Comparison plot disimpan: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def run_encoder_comparison(
    cfg,
    train_dl: DataLoader,
    val_dl: DataLoader,
    label_enc,
    edge_index: Tensor,
    leaf_node_ids=None,
    n_epochs: int = 20,
) -> "pd.DataFrame":
    """
    Jalankan training dua kali (mode 'pretrain' dan 'scratch') lalu
    simpan metrik ke CSV terpisah dan tampilkan comparison plot.

    Hanya valid jika cfg.model.encoder_mode == 'all'.

    Output CSV:
      {metric_dir}/{experiment_name}_pretrain_train_log.csv
      {metric_dir}/{experiment_name}_scratch_train_log.csv

    Output plot:
      {plot_dir}/{experiment_name}_encoder_comparison.png

    Parameter:
        cfg           : Config object dengan encoder_mode='all'
        train_dl      : DataLoader training
        val_dl        : DataLoader validasi
        label_enc     : HierarchicalLabelEncoder
        edge_index    : taxonomy graph edges
        leaf_node_ids : Species-level leaf node IDs
        n_epochs      : jumlah epoch per run (default 20)

    Return:
        DataFrame perbandingan best F1 dengan kolom:
          mode, best_val_f1_macro, best_val_f1_species, best_val_f1_genus, best_epoch
    """
    import copy
    import pandas as pd
    from pathlib import Path as _Path

    if cfg.model.encoder_mode != "all":
        raise ValueError(
            f"run_encoder_comparison() hanya untuk encoder_mode='all', "
            f"bukan '{cfg.model.encoder_mode}'."
        )

    from taxograph_bert import build_model
    from trainer import Trainer

    plot_dir = _Path(cfg.paths.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for mode in ("pretrain", "scratch"):
        logger.info(f"[encoder_comparison] Memulai run: mode='{mode}' ({n_epochs} epochs)")

        cfg_run = copy.deepcopy(cfg)
        cfg_run.model.encoder_mode        = mode
        cfg_run.train.max_epochs          = n_epochs
        cfg_run.train.early_stopping_patience = n_epochs  # nonaktifkan early stop

        try:
            model = build_model(cfg_run, label_enc, edge_index, leaf_node_ids)
        except RuntimeError as exc:
            logger.error(f"[encoder_comparison] Gagal build model mode='{mode}': {exc}")
            continue

        trainer = Trainer(model, train_dl, val_dl, cfg_run, run_tag=mode)
        trainer.setup_optimizers()
        trainer.train()

        # Baca CSV hasil training
        csv_path = _Path(cfg_run.paths.metric_dir) / f"{cfg_run.experiment_name}_{mode}_train_log.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df_val = df[df["phase"] == "val"]
            if len(df_val):
                best_row = df_val.loc[df_val["f1_macro"].idxmax()]
                results.append({
                    "mode":                mode,
                    "best_val_f1_macro":   float(best_row["f1_macro"]),
                    "best_val_f1_species": float(best_row.get("f1_Species", 0.0)),
                    "best_val_f1_genus":   float(best_row.get("f1_Genus", 0.0)),
                    "best_epoch":          int(best_row["epoch"]),
                })

    # Plot jika kedua CSV tersedia
    pretrain_csv = _Path(cfg.paths.metric_dir) / f"{cfg.experiment_name}_pretrain_train_log.csv"
    scratch_csv  = _Path(cfg.paths.metric_dir) / f"{cfg.experiment_name}_scratch_train_log.csv"
    if pretrain_csv.exists() and scratch_csv.exists():
        plot_encoder_comparison(
            pretrain_csv,
            scratch_csv,
            save_path=plot_dir / f"{cfg.experiment_name}_encoder_comparison.png",
        )

    summary_df = pd.DataFrame(results)
    logger.info(f"[encoder_comparison] Selesai.\n{summary_df.to_string(index=False)}")
    return summary_df


@torch.no_grad()
def _collect_ood_scores_from_fasta(
    model, tokenizer, fasta_path: str | Path, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """Kumpulkan skor OOD dari file FASTA secara streaming (tidak load semua sekuens ke RAM)."""
    try:
        from Bio import SeqIO
    except ImportError:
        logger.error("Biopython diperlukan: pip install biopython")
        return np.array([])

    scores_list = []
    batch_seqs: list[str] = []

    def _flush(seqs: list[str]) -> None:
        """Forward pass satu batch dan simpan skor OOD."""
        enc = tokenizer.batch_encode(seqs)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids, attention_mask, update_graph=False)
        scores_list.append(out["ood_score"].cpu().numpy())
        del input_ids, attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Streaming: SeqIO.parse() adalah generator — tidak load seluruh file ke RAM
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        batch_seqs.append(str(record.seq).upper())
        if len(batch_seqs) >= batch_size:
            _flush(batch_seqs)
            batch_seqs = []

    if batch_seqs:  # flush sisa batch terakhir
        _flush(batch_seqs)

    return np.concatenate(scores_list) if scores_list else np.array([])


def _delong_ci_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Hitung confidence interval AUROC dengan bootstrap (DeLong approximation).
    Mengembalikan (ci_lo, ci_hi) pada level kepercayaan 1-alpha.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    aucs = []

    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except Exception:
            pass

    aucs = np.array(aucs)
    lo   = float(np.percentile(aucs, 100 * alpha / 2))
    hi   = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def _compute_fpr_at_tpr(
    y_true: np.ndarray, y_score: np.ndarray, tpr_target: float = 0.95
) -> float:
    """Hitung FPR pada TPR tertentu (tpr_target) dari ROC curve."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Cari FPR di titik TPR >= tpr_target
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def evaluate_with_silva_classifier(
    qza_path,
    val_sequences,
    val_labels,
    label_enc,
    cfg,
    csv_path=None,
    device: str = "cpu",
):
    """
    Evaluasi SILVA Naive Bayes classifier dari file .qza sebagai baseline perbandingan.

    File .qza adalah arsip ZIP yang berisi sklearn_pipeline.pkl (sklearn Pipeline).
    Pipeline berisi TF-IDF k-mer vectorizer + Naive Bayes classifier.

    Alasan digunakan: QIIME2's SILVA classifier adalah baseline standar komunitas
    untuk klasifikasi 16S rRNA. Membandingkan TaxoGraph-BERT vs ini memberikan
    konteks performa riset yang kuat.

    Parameter:
        qza_path       : path ke file .qza SILVA classifier
        val_sequences  : list sekuens DNA (string) dari validation set
        val_labels     : (N, 6) int array — ground truth label indices
        label_enc      : HierarchicalLabelEncoder untuk decode predictions
        cfg            : Config object
        csv_path       : opsional — append hasil ke CSV; jika None gunakan cfg.paths.metric_dir
        device         : tidak dipakai (classifier sklearn berjalan di CPU)

    Return:
        DataFrame dengan kolom: rank, f1_macro, n_samples, model
        atau None jika .qza tidak ditemukan / ekstraksi gagal
    """
    import zipfile
    import tempfile
    import pickle
    import pandas as pd
    import numpy as np
    from pathlib import Path as _Path
    from sklearn.metrics import f1_score

    qza_path = _Path(qza_path)
    if not qza_path.exists():
        logger.warning(f"[silva_baseline] File .qza tidak ditemukan: {qza_path}")
        logger.info(
            "[silva_baseline] Download SILVA classifier dari: "
            "https://data.qiime2.org/2023.5/common/silva-138-99-seqs-515-806-nb-classifier.qza"
        )
        return None

    # Ekstrak classifier dari arsip .qza (format: ZIP)
    classifier = None
    try:
        with zipfile.ZipFile(str(qza_path), 'r') as zf:
            pkl_files = [
                n for n in zf.namelist()
                if n.endswith('sklearn_pipeline.pkl') or n.endswith('classifier.pkl')
            ]
            if not pkl_files:
                logger.error(f"[silva_baseline] Tidak ada classifier.pkl di dalam {qza_path.name}")
                return None
            with tempfile.TemporaryDirectory() as tmpdir:
                extracted = zf.extract(pkl_files[0], path=tmpdir)
                with open(extracted, 'rb') as f:
                    classifier = pickle.load(f)
                logger.info(f"[silva_baseline] Classifier dimuat dari {pkl_files[0]}")
    except Exception as e:
        logger.error(f"[silva_baseline] Gagal mengekstrak .qza: {e}")
        return None

    RANK_NAMES = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]
    rows = []

    # SILVA NB classifier memprediksi full taxonomy string, bukan per-rank
    # Contoh output: "d__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; ..."
    try:
        predictions = classifier.predict(val_sequences)
    except Exception as e:
        logger.error(f"[silva_baseline] Gagal predict: {e}")
        return None

    rank_prefix_map = {
        "Phylum": "p__", "Class": "c__", "Order": "o__",
        "Family": "f__", "Genus": "g__", "Species": "s__",
    }
    for r_idx, rank in enumerate(RANK_NAMES):
        y_true_indices = np.array(val_labels)[:, r_idx]
        valid_mask = y_true_indices != 0  # skip PAD

        if valid_mask.sum() == 0:
            continue

        rank_prefix = rank_prefix_map[rank]
        pred_labels_str = []
        for pred_str in predictions:
            parts = str(pred_str).split("; ")
            rank_val = next(
                (p.split(rank_prefix)[-1] for p in parts if rank_prefix in p),
                "Unknown"
            )
            pred_labels_str.append(rank_val)

        # Encode ke indices via label_enc
        try:
            pred_indices = np.array([
                label_enc.label2idx.get(rank, {}).get(lbl, 0)
                for lbl in pred_labels_str
            ])
        except Exception:
            pred_indices = np.zeros(len(pred_labels_str), dtype=int)

        y_true_v = y_true_indices[valid_mask]
        y_pred_v = pred_indices[valid_mask]

        f1 = f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)
        rows.append({
            "rank":      rank,
            "f1_macro":  float(f1),
            "n_samples": int(valid_mask.sum()),
            "model":     "silva_naive_bayes",
        })

    df = pd.DataFrame(rows)

    # Append ke CSV log jika diminta
    if csv_path is not None or cfg is not None:
        _csv = (
            _Path(csv_path) if csv_path
            else _Path(cfg.paths.metric_dir) / f"{cfg.experiment_name}_silva_baseline.csv"
        )
        df.to_csv(_csv, index=False, mode='a', header=not _csv.exists())
        logger.info(f"[silva_baseline] Hasil disimpan ke: {_csv}")

    logger.info(f"[silva_baseline] SILVA NB Classifier baseline:\n{df.to_string(index=False)}")
    return df
