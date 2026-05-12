"""
training/trainer.py — Training Engine TaxoGraph-BERT
=====================================================
Modul ini mengelola seluruh siklus pelatihan:
  Phase 1: Pre-training HyTaxGNN (Poincaré embedding via Riemannian SGD)
  Phase 2: Joint fine-tuning end-to-end dengan Hierarchical CE Loss

Fitur:
  - tqdm progress bar per batch dan per epoch
  - AMP (Automatic Mixed Precision) via torch.cuda.amp
  - Gradient accumulation untuk effective batch besar
  - Early stopping berbasis val Hierarchical F1
  - Checkpoint otomatis (best + latest) ke format PKL/PT
  - Backup metrik per epoch ke CSV

Penggunaan:
    from training.trainer import Trainer
    trainer = Trainer(model, train_dl, val_dl, CFG)
    trainer.train()
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast  # torch.cuda.amp deprecated since PyTorch 2.4
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm.auto import tqdm
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CROSS-ENTROPY LOSS
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalCrossEntropyLoss(nn.Module):
    """
    Hierarchical Cross-Entropy Loss dengan penalti eksponensial lintas rank.

    Formula:
      L = Σ_r w_r · CE(ŷ_r, y_r) + γ · Σ_r Σ_{r'>r} exp(rank_gap) · 1[ŷ_r ≠ y_r]

    - w_r       : bobot per rank (lebih besar untuk rank lebih halus)
    - γ         : skala penalti lintas rank
    - rank_gap  : jarak antar rank yang salah (1=dalam order, 5=lintas phylum)
    - label_smoothing: mencegah overconfidence

    Parameter:
        rank_weights    : list bobot per rank (Phylum → Species)
        gamma           : skala penalti cross-rank
        label_smoothing : epsilon label smoothing
        ignore_index    : index label yang diabaikan (0 = PAD)

    Input:
        logits_list : list of 6 tensor (B, C_rank) — logits per rank
        labels      : (B, 6) int64 — ground truth index per rank

    Return:
        scalar loss tensor
    """

    def __init__(
        self,
        rank_weights: List[float] = None,
        gamma: float = 0.1,
        label_smoothing: float = 0.05,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.rank_weights    = rank_weights or [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index    = ignore_index
        self.n_ranks         = len(self.rank_weights)

        # CE loss per rank (tanpa reduction agar bisa di-mask)
        self.ce_losses = nn.ModuleList([
            nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction="mean",
            )
            for _ in range(self.n_ranks)
        ])

    def forward(
        self, logits_list: List[Tensor], labels: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Return:
            (total_loss, loss_dict) — loss total dan breakdown per rank
        """
        total_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
        loss_dict  = {}

        for r, (logits, w, ce_fn) in enumerate(
            zip(logits_list, self.rank_weights, self.ce_losses)
        ):
            y_r = labels[:, r]  # (B,)

            # CE loss untuk rank ini
            loss_r = w * ce_fn(logits, y_r)

            # Penalti cross-rank: jika salah di rank r, cek apakah prediksi
            # juga salah di rank yang lebih kasar (indikatif kesalahan parah)
            pred_r = logits.argmax(dim=-1)  # (B,)
            wrong  = (pred_r != y_r) & (y_r != self.ignore_index)  # (B,) bool

            if wrong.any() and r > 0:
                # Penalti hierarkis: kesalahan di rank LEBIH KASAR lebih dipenalti
                # rank_distance = jarak dari rank tertinggi (Phylum=0) ke rank saat ini
                # Phylum error (r=0) → exp(n_ranks-0) = max penalty
                # Species error (r=5) → exp(n_ranks-5) = min penalty
                rank_distance  = self.n_ranks - r  # lebih besar = rank lebih kasar
                cross_penalty  = self.gamma * torch.exp(
                    torch.tensor(float(rank_distance))
                ) * wrong.float().mean()
                loss_r = loss_r + cross_penalty

            total_loss = total_loss + loss_r
            loss_dict[f"loss_rank_{r}"] = loss_r.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Engine pelatihan penuh untuk TaxoGraph-BERT.

    Parameter:
        model    : TaxoGraphBERT instance
        train_dl : DataLoader training
        val_dl   : DataLoader validation
        cfg      : Config object

    Penggunaan:
        trainer = Trainer(model, train_dl, val_dl, cfg)
        trainer.phase1_pretrain_hgnn()   # Phase 1 (optional jika sudah punya checkpoint)
        trainer.train()                  # Phase 2: joint fine-tuning
    """

    RANK_NAMES = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]

    def __init__(self, model, train_dl, val_dl, cfg, run_tag: str = ""):
        self.model    = model
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.cfg      = cfg
        self.device   = torch.device(cfg.device)  # langsung dari cfg (mendukung mps/cpu/cuda)
        self._run_tag = run_tag

    @property
    def _raw_model(self):
        """Kembalikan model tanpa nn.DataParallel wrapper (untuk akses buffer & submodul)."""
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        # ── Paths ──────────────────────────────────────────────────────
        self.ckpt_dir   = Path(cfg.paths.checkpoint_dir)
        self.metric_dir = Path(cfg.paths.metric_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metric_dir.mkdir(parents=True, exist_ok=True)

        # ── Loss ───────────────────────────────────────────────────────
        self.criterion = HierarchicalCrossEntropyLoss(
            rank_weights=cfg.train.rank_weights,
            gamma=cfg.train.hierarchical_penalty_gamma,
            label_smoothing=cfg.train.label_smoothing,
        )

        # ── Optimizers (diinisialisasi di _build_optimizers) ──────────
        self.optimizer   = None
        self.scheduler   = None
        # GradScaler hanya efektif di CUDA; CPU/MPS gunakan dummy (enabled=False)
        _amp_device = self.device.type if self.device.type == 'cuda' else 'cpu'
        _amp_enabled = cfg.train.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler(_amp_device, enabled=_amp_enabled)

        # ── State ──────────────────────────────────────────────────────
        self.best_val_f1   = 0.0
        self.patience_ctr  = 0
        self.global_step   = 0
        self.epoch_metrics: List[Dict] = []

        # ── CSV backup ─────────────────────────────────────────────────
        _suffix = f"_{run_tag}" if run_tag else ""
        self.csv_path = self.metric_dir / f"{cfg.experiment_name}{_suffix}_train_log.csv"
        self._init_csv()

    def _init_csv(self, append: bool = False):
        """Inisialisasi file CSV untuk backup metrik per epoch.

        Parameter:
            append : jika True dan file sudah ada, pertahankan isinya (resume mode).
                     jika False (default), buat ulang file dari awal.
        """
        if append and self.csv_path.exists():
            logger.info(f"CSV log (resume/append): {self.csv_path}")
            return  # pertahankan file yang sudah ada
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "phase", "loss_total",
                *[f"loss_rank_{r}" for r in range(6)],
                *[f"f1_{rank}" for rank in self.RANK_NAMES],
                "f1_macro", "lr", "elapsed_sec",
            ])
            writer.writeheader()
        logger.info(f"CSV log: {self.csv_path}")

    def _append_csv(self, row: Dict):
        """Tambahkan satu baris ke CSV backup."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    # ── PHASE 1: Pre-training HyTaxGNN ────────────────────────────────────
    def phase1_pretrain_hgnn(self):
        """
        Phase 1: Latih hanya HyTaxGNN dengan Poincaré embedding loss.
        Gunakan Riemannian SGD dari geoopt untuk optimisasi di manifold.

        Target: node embedding yang mencerminkan jarak evolusi biologis
        sebelum joint fine-tuning.
        """
        logger.info("=== PHASE 1: Pre-training HyTaxGNN ===")

        try:
            import geoopt
            optimizer_p1 = geoopt.optim.RiemannianSGD(
                self._raw_model.hytaxgnn.parameters(),
                lr=self.cfg.train.pretrain_hgnn_lr,
                stabilize=10,
            )
        except ImportError:
            logger.warning("geoopt tidak terinstal, fallback ke AdamW untuk Phase 1.")
            optimizer_p1 = AdamW(
                self._raw_model.hytaxgnn.parameters(),
                lr=self.cfg.train.pretrain_hgnn_lr,
            )

        # Loss Phase 1: Poincaré embedding reconstruction loss
        # (jarak hiperbolik antara parent-child harus kecil, antar-phylum harus besar)
        n_epochs = self.cfg.train.pretrain_hgnn_epochs

        # ── Mitigasi VRAM: offload submodul non-HyTaxGNN ke CPU selama Phase 1 ──
        # CUDA: seq_encoder (~6-8 GB) dikeluarkan dari VRAM agar Phase 1 tidak OOM.
        # CPU/MPS: tidak perlu offload — jalankan langsung di device asal.
        is_cuda = self.device.type == 'cuda'
        _offloaded_mods = []
        try:
            if is_cuda:
                for name in ('seq_encoder', 'fusion', 'classifier', 'ood_head'):
                    submod = getattr(self._raw_model, name)
                    submod.cpu()
                    _offloaded_mods.append((name, submod))
                torch.cuda.empty_cache()
                logger.info("Phase 1: submodul non-HyTaxGNN dioffload ke CPU (VRAM freed).")

            edge_batch_size = getattr(self.cfg.train, 'phase1_edge_batch_size', 4096)
            edge_index_p1   = self._raw_model.edge_index  # sudah ada di device
            E               = edge_index_p1.shape[1]

            pbar = tqdm(range(n_epochs), desc="Phase 1 — HyTaxGNN Pre-train", unit="epoch")
            for epoch in pbar:
                self._raw_model.hytaxgnn.train()
                optimizer_p1.zero_grad()

                # Dapatkan semua node embeddings
                node_emb = self._raw_model.hytaxgnn(edge_index_p1)  # (N, d)

                # Mini-batch edges: hindari spike memori saat E sangat besar
                perm    = torch.randperm(E, device=self.device)[:min(edge_batch_size, E)]
                src_b   = edge_index_p1[0][perm]
                dst_b   = edge_index_p1[1][perm]
                B_e     = src_b.shape[0]

                # Jarak parent-child (harus kecil)
                d_pos = self._raw_model.ball.dist(node_emb[src_b], node_emb[dst_b]).mean()

                # Jarak random pairs (harus lebih besar)
                rnd_idx = torch.randint(0, node_emb.shape[0], (B_e,), device=self.device)
                d_neg   = self._raw_model.ball.dist(node_emb[src_b], node_emb[rnd_idx]).mean()

                # Margin-based loss: d_pos < d_neg - margin
                margin = 0.5
                loss   = F.relu(d_pos - d_neg + margin)

                loss.backward()
                optimizer_p1.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "d_pos": f"{d_pos.item():.3f}",
                                  "d_neg": f"{d_neg.item():.3f}"})

            # Simpan checkpoint Phase 1
            ckpt_p1 = self.ckpt_dir / "phase1_hgnn.pt"
            torch.save({"hgnn_state": self._raw_model.hytaxgnn.state_dict(), "epoch": n_epochs}, ckpt_p1)
            logger.info(f"Phase 1 selesai. Checkpoint: {ckpt_p1}")

        finally:
            # Selalu kembalikan submodul ke device semula (bahkan jika terjadi error)
            if _offloaded_mods:
                for name, submod in _offloaded_mods:
                    submod.to(self.device)
                if is_cuda:
                    torch.cuda.empty_cache()
                logger.info("Phase 1: submodul dikembalikan ke device semula.")

    # ── PHASE 2: Joint Fine-tuning ────────────────────────────────────────
    def _build_optimizers(self):
        """
        Bangun optimizer terpisah untuk Transformer backbone dan komponen lain.
        Learning rate berbeda: Transformer lebih kecil (3e-5) karena sudah pre-trained.
        """
        tcfg = self.cfg.train
        mcfg = self.cfg.model

        # Kelompokkan parameter berdasarkan komponen
        transformer_params = list(self._raw_model.seq_encoder.parameters())
        other_params = (
            list(self._raw_model.hytaxgnn.parameters())
            + list(self._raw_model.fusion.parameters())
            + list(self._raw_model.classifier.parameters())
            + list(self._raw_model.ood_head.parameters())
        )

        self.optimizer = AdamW([
            {"params": transformer_params, "lr": tcfg.lr_transformer,
             "weight_decay": tcfg.weight_decay_transformer},
            {"params": other_params,       "lr": tcfg.lr_classifier,
             "weight_decay": 1e-4},
        ])

        # Scheduler: linear warmup → cosine annealing
        total_steps   = len(self.train_dl) * tcfg.max_epochs // tcfg.gradient_accumulation_steps
        warmup_steps  = int(total_steps * tcfg.warmup_ratio)

        warmup_sched  = LinearLR(self.optimizer, start_factor=0.1,
                                  end_factor=1.0, total_iters=warmup_steps)
        cosine_sched  = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=total_steps - warmup_steps, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_steps]
        )

        logger.info(f"Optimizer: {len(transformer_params)} Transformer params "
                    f"(lr={tcfg.lr_transformer}) + "
                    f"{len(other_params)} other params (lr={tcfg.lr_classifier})")

    def setup_optimizers(self):
        """Public alias untuk _build_optimizers — kompatibilitas notebook."""
        self._build_optimizers()

    def train(self, start_epoch: Optional[int] = None):
        """
        Main training loop — Phase 2 joint fine-tuning.
        Jalankan setelah phase1_pretrain_hgnn().

        Parameter:
            start_epoch : epoch terakhir yang sudah selesai.
                          Jika None (default), auto-detect dari checkpoint terbaru.
                          Pass 0 untuk memaksa mulai dari awal.
        """
        logger.info("=== PHASE 2: Joint Fine-tuning ===")
        if self.optimizer is None:
            self._build_optimizers()
        tcfg = self.cfg.train

        # ── Auto-resume ───────────────────────────────────────────────
        if start_epoch is None:
            start_epoch = self.resume_from_latest()

        is_resume = start_epoch > 0
        if is_resume:
            # Jangan overwrite CSV epoch sebelumnya
            self._init_csv(append=True)
            logger.info(f"Training dilanjutkan dari epoch {start_epoch + 1} / {tcfg.max_epochs}")
        else:
            # Fresh start — tulis ulang header CSV
            self._init_csv(append=False)

        self._training_complete = False

        epoch_pbar = tqdm(range(start_epoch + 1, tcfg.max_epochs + 1),
                          desc="Epoch", unit="epoch", position=0, leave=True)

        for epoch in epoch_pbar:
            t_start = time.time()

            # Training
            train_metrics = self._train_one_epoch(epoch)

            # Validation
            val_metrics   = self._validate(epoch)

            elapsed = time.time() - t_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Baris train terpisah
            train_row = {
                "epoch":      epoch,
                "phase":      "train",
                "loss_total": train_metrics.get("loss_total", 0),
                **{f"loss_rank_{r}": train_metrics.get(f"loss_rank_{r}", 0) for r in range(6)},
                **{f"f1_{rn}": 0.0 for rn in self.RANK_NAMES},  # F1 tidak dihitung di train
                "f1_macro":   0.0,
                "lr":         current_lr,
                "elapsed_sec": elapsed,
            }
            # Baris val terpisah
            val_row = {
                "epoch":      epoch,
                "phase":      "val",
                "loss_total": 0.0,
                **{f"loss_rank_{r}": 0.0 for r in range(6)},
                **{f"f1_{rn}": val_metrics.get(f"f1_{rn}", 0) for rn in self.RANK_NAMES},
                "f1_macro":   val_metrics.get("f1_macro", 0),
                "lr":         current_lr,
                "elapsed_sec": elapsed,
            }
            self._append_csv(train_row)
            self._append_csv(val_row)
            self.epoch_metrics.append(train_row)
            self.epoch_metrics.append(val_row)

            # Update progress bar
            epoch_pbar.set_postfix({
                "val_f1_genus":   f"{val_metrics.get('f1_Genus', 0):.4f}",
                "val_f1_species": f"{val_metrics.get('f1_Species', 0):.4f}",
                "loss":           f"{train_metrics.get('loss_total', 0):.4f}",
            })

            # Early stopping & checkpointing
            val_f1 = val_metrics.get("f1_macro", 0)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_ctr = 0
                self._save_checkpoint(epoch, "best")
                logger.info(f"Epoch {epoch}: NEW BEST val F1 = {val_f1:.4f} → checkpoint disimpan")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= tcfg.early_stopping_patience:
                    logger.info(f"Early stopping triggered setelah {epoch} epoch.")
                    break

            # Selalu simpan checkpoint terbaru
            self._save_checkpoint(epoch, "latest")

        # Tandai training selesai (digunakan saat simpan _latest checkpoint terakhir)
        self._training_complete = True
        # Kalibrasi OOD head setelah training selesai
        self._calibrate_ood()

        logger.success(f"Training selesai. Best val macro F1: {self.best_val_f1:.4f}")
        return self.epoch_metrics

    def _train_one_epoch(self, epoch: int) -> Dict:
        """
        Satu epoch training dengan gradient accumulation dan AMP.

        Return:
            dict loss metrics rata-rata epoch ini
        """
        self.model.train()
        tcfg = self.cfg.train
        accumulate = tcfg.gradient_accumulation_steps

        total_loss  = 0.0
        rank_losses = {f"loss_rank_{r}": 0.0 for r in range(6)}
        n_batches   = 0

        pbar = tqdm(
            enumerate(self.train_dl),
            total=len(self.train_dl),
            desc=f"  Train Epoch {epoch}",
            leave=False,
            unit="batch",
            position=1,
        )

        self.optimizer.zero_grad()

        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            # Forward pass dengan AMP (device-aware: hanya aktif di CUDA)
            _amp_dt = self.device.type if self.device.type in ('cuda', 'cpu') else 'cpu'
            _amp_en = tcfg.use_amp and self.device.type == 'cuda'
            with autocast(device_type=_amp_dt, enabled=_amp_en):
                out    = self.model(input_ids, attention_mask)
                logits = out["logits"]
                loss, loss_dict = self.criterion(logits, labels)
                loss = loss / accumulate  # normalize untuk gradient accumulation

            # Backward — GradScaler hanya untuk CUDA; CPU/MPS langsung backward
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update ogni `accumulate` steps
            if (step + 1) % accumulate == 0:
                if self.device.type == 'cuda':
                    # Gradient clipping via scaler (CUDA)
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), tcfg.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping tanpa scaler (CPU / MPS)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), tcfg.max_grad_norm
                    )
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Akumulasi metrik
            total_loss += loss_dict["loss_total"]
            for r in range(6):
                rank_losses[f"loss_rank_{r}"] += loss_dict.get(f"loss_rank_{r}", 0)
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss_dict['loss_total']:.4f}"})

        # Rata-rata
        avg = {"loss_total": total_loss / max(n_batches, 1)}
        avg.update({k: v / max(n_batches, 1) for k, v in rank_losses.items()})
        return avg

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict:
        """
        Validasi model pada val_dl, hitung F1-Score per rank.

        Return:
            dict F1 per rank + macro F1
        """
        from sklearn.metrics import f1_score

        self.model.eval()

        all_preds  = [[] for _ in range(6)]  # prediksi per rank
        all_labels = [[] for _ in range(6)]  # label benar per rank

        pbar = tqdm(
            self.val_dl,
            desc=f"  Val   Epoch {epoch}",
            leave=False,
            unit="batch",
            position=1,
        )

        for batch in pbar:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"]  # tetap di CPU

            out    = self.model(input_ids, attention_mask, update_graph=False)
            logits = out["logits"]

            for r in range(6):
                preds_r = logits[r].argmax(dim=-1).cpu().numpy()
                labs_r  = labels[:, r].numpy()
                # Abaikan PAD label (0)
                mask    = labs_r != 0
                all_preds[r].extend(preds_r[mask].tolist())
                all_labels[r].extend(labs_r[mask].tolist())

        # Hitung F1 per rank
        metrics = {}
        f1_values = []
        for r, rname in enumerate(self.RANK_NAMES):
            if len(all_labels[r]) == 0:
                metrics[f"f1_{rname}"] = 0.0
                continue
            f1 = f1_score(
                all_labels[r], all_preds[r],
                average="macro", zero_division=0
            )
            metrics[f"f1_{rname}"] = f1
            f1_values.append(f1)
            logger.debug(f"    F1 {rname}: {f1:.4f}")

        metrics["f1_macro"] = float(np.mean(f1_values)) if f1_values else 0.0
        return metrics

    # ── Checkpointing ─────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, tag: str):
        """
        Simpan model checkpoint ke file .pt (PyTorch format).

        File yang disimpan:
          outputs/checkpoints/{experiment_name}_{tag}.pt
        Berisi: model state_dict, optimizer state, epoch, best_val_f1
        """
        ckpt = {
            "epoch":              epoch,
            "model_state":        self._raw_model.state_dict(),
            "optimizer":          self.optimizer.state_dict(),
            "scheduler":          self.scheduler.state_dict() if self.scheduler else None,
            "scaler":             self.scaler.state_dict(),
            "best_val_f1":        self.best_val_f1,
            "val_f1_macro":       self.best_val_f1,  # alias agar notebook Phase 2 cell tidak KeyError
            "patience_ctr":       self.patience_ctr,
            "global_step":        self.global_step,
            "training_complete":  tag == "latest" and getattr(self, "_training_complete", False),
            "cfg":                self.cfg,
        }
        path = self.ckpt_dir / f"{self.cfg.experiment_name}{'_' + self._run_tag if self._run_tag else ''}_{tag}.pt"
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        """
        Muat checkpoint dari file .pt.

        Parameter:
            path : path file checkpoint .pt
        """
        ckpt = torch.load(path, map_location=self.device)
        self._raw_model.load_state_dict(ckpt["model_state"])
        if self.optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self.best_val_f1  = ckpt.get("best_val_f1", 0.0)
        self.patience_ctr = ckpt.get("patience_ctr", 0)
        self.global_step  = ckpt.get("global_step", 0)
        logger.info(f"Checkpoint dimuat: {path} (epoch={ckpt['epoch']}, "
                    f"best_f1={self.best_val_f1:.4f})")

    def resume_from_latest(self) -> int:
        """
        Cari checkpoint *_latest.pt dan muat semua state untuk melanjutkan training.

        Return:
            Epoch terakhir yang selesai (0 jika tidak ada checkpoint / fresh start).

        Perilaku:
            - Jika latest checkpoint ada dan training_complete=True → return epoch (sudah selesai)
            - Jika latest checkpoint ada dan training_complete=False → restore state, return epoch
            - Jika tidak ada checkpoint → return 0 (fresh start)
        """
        import glob as _glob
        _suffix = f"_{self._run_tag}" if self._run_tag else ""
        pattern = str(self.ckpt_dir / f"{self.cfg.experiment_name}{_suffix}_latest.pt")
        matches = sorted(_glob.glob(pattern))

        if not matches:
            logger.info("Resume: tidak ada checkpoint ditemukan — mulai dari awal.")
            return 0

        ckpt_path = matches[-1]
        ckpt = torch.load(ckpt_path, map_location=self.device)
        last_epoch = ckpt.get("epoch", 0)

        if ckpt.get("training_complete", False):
            logger.info(f"Resume: training sudah selesai di epoch {last_epoch} — skip training.")
            return last_epoch

        # Restore model state
        self._raw_model.load_state_dict(ckpt["model_state"])

        # Bangun optimizer/scheduler dulu jika belum ada, lalu restore state
        if self.optimizer is None:
            self._build_optimizers()
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])

        self.best_val_f1  = ckpt.get("best_val_f1", 0.0)
        self.patience_ctr = ckpt.get("patience_ctr", 0)
        self.global_step  = ckpt.get("global_step", 0)

        logger.info(
            f"Resume dari epoch {last_epoch} | best_f1={self.best_val_f1:.4f} | "
            f"patience={self.patience_ctr} | global_step={self.global_step}"
        )
        return last_epoch

    @torch.no_grad()
    def _calibrate_ood(self):
        """
        Kalibrasi threshold OOD head menggunakan val_dl setelah training selesai.
        Mengumpulkan hyp_emb dari seluruh val set, hitung OOD scores,
        dan panggil model.ood_head.calibrate() untuk menetapkan threshold delta.
        """
        logger.info("Mengkalibrasi OOD threshold dari validation set...")
        self.model.eval()

        node_emb = self._raw_model.get_node_embeddings().to(self.device)
        leaf_emb = node_emb[self._raw_model.leaf_node_ids]

        all_scores = []
        for batch in tqdm(self.val_dl, desc="OOD calibration", leave=False):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            out    = self.model(input_ids, attention_mask, update_graph=False)
            z_hyp  = out["hyp_emb"]  # (B, d)
            scores = self._raw_model.ood_head.compute_ood_score(z_hyp, leaf_emb)  # (B,)
            all_scores.append(scores.cpu())

        if all_scores:
            all_scores_tensor = torch.cat(all_scores, dim=0)
            self._raw_model.ood_head.calibrate(all_scores_tensor)
            logger.info(f"OOD threshold dikalibrasi: delta = {self._raw_model.ood_head.threshold.item():.4f}")
        else:
            logger.warning("Val set kosong; OOD threshold tidak dikalibrasi.")