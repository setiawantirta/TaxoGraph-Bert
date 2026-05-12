"""
models/taxograph_bert.py — Model Lengkap TaxoGraph-BERT
========================================================
Menggabungkan semua komponen menjadi model end-to-end:
  1. Transformer Sequence Encoder (DNABERT-2 + LoRA)
  2. HyTaxGNN (Hyperbolic Taxonomy GNN)
  3. Feature Fusion (Möbius Addition)
  4. Hierarchical Classification Head
  5. OOD Detection Head (hyperbolic margin scoring)

Penggunaan:
    from models.taxograph_bert import TaxoGraphBERT, build_model
    model = build_model(CFG, label_encoder, edge_index)
    out = model(input_ids, attention_mask, edge_index)
    # out["logits"]    : list of len-6 tensor (per rank)
    # out["ood_score"] : tensor (B,) — skor OOD (lebih besar = lebih OOD)
    # out["hyp_emb"]   : tensor (B, poincare_dim) — query di Poincaré space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from hyperbolic import HyTaxGNN, PoincareBall


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSFORMER SEQUENCE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class DNASequenceEncoder(nn.Module):
    """
    Sequence encoder berbasis DNABERT-2 dengan LoRA fine-tuning.
    Jika DNABERT-2 tidak tersedia (offline), gunakan encoder sederhana berbasis
    multi-layer Transformer sebagai fallback.

    Strategi data besar:
      - LoRA hanya melatih ~5% parameter (rank=16) → memori lebih hemat
      - Gradient checkpointing dapat diaktifkan untuk menghemat VRAM

    Parameter:
        backbone   : HuggingFace model ID atau "scratch" untuk fallback
        hidden_dim : dimensi hidden layer (768 untuk DNABERT-2)
        pooling    : "mean" | "cls" | "max" — cara aggregate token embeddings
        use_lora   : apakah terapkan LoRA
        lora_r     : rank LoRA
        lora_alpha : scaling alpha LoRA
        freeze_base: apakah bekukan seluruh base model (hanya latih LoRA)

    Input:
        input_ids      : (B, L) token IDs
        attention_mask : (B, L) 1/0 mask

    Output:
        (B, hidden_dim) — sequence embeddings
    """

    def __init__(
        self,
        backbone: str = "zhihan1996/DNABERT-2-117M",
        hidden_dim: int = 768,
        pooling: str = "mean",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        vocab_size: int = 4100,   # 4096 k-mer + 4 special tokens
        fallback_layers: int = 6, # jika pakai encoder scratch
        encoder_mode: str = "pretrain",  # "pretrain" | "scratch" | "all"
    ):
        super().__init__()
        self.pooling    = pooling
        self.hidden_dim = hidden_dim
        self._using_pretrained = False

        if encoder_mode == "scratch":
            # Mode scratch: langsung buat encoder dari awal, tanpa mencoba DNABERT-2
            print(f"[DNASequenceEncoder] Mode 'scratch': encoder dilatih dari awal "
                  f"(vocab={vocab_size}, d={hidden_dim}, layers={fallback_layers}).")
            self.encoder = self._build_scratch_encoder(vocab_size, hidden_dim, fallback_layers)
        else:
            # Mode "pretrain" atau "all": wajib menggunakan DNABERT-2
            # Fallback ke scratch TIDAK diizinkan — error harus eksplisit
            try:
                from transformers import AutoModel, AutoConfig
                from peft import get_peft_model, LoraConfig, TaskType
                import math as _math
                import sys as _sys
                import types as _types

                config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
                if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
                    config.pad_token_id = 0

                # ── FASE 1: Download + import bert_layers, lalu patch BertEncoder ─
                # Root cause: di dalam accelerate's init_empty_weights() context,
                # torch.arange(device=None) → meta, TETAPI torch.Tensor(python_list)
                # SELALU → CPU (konstruktor lama mengabaikan context). Operasi
                # slopes * -relative_position (CPU × meta) → RuntimeError.
                # Solusi: import bert_layers lebih dulu via dynamic_module_utils,
                # patch BertEncoder.rebuild_alibi_tensor di kelas, lalu panggil
                # from_pretrained — yang menggunakan kelas ter-patch dari sys.modules.
                _BertEncoder = None
                try:
                    from transformers.dynamic_module_utils import (
                        get_class_from_dynamic_module as _gcfdm,
                    )
                    # Picu download + import bert_layers.py ke sys.modules
                    try:
                        _gcfdm("bert_layers.BertModel", backbone, trust_remote_code=True)
                    except TypeError:
                        try:
                            _gcfdm("bert_layers.BertModel", backbone)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Cari BertEncoder di semua modul yang baru diimpor
                    for _mk, _mv in list(_sys.modules.items()):
                        if (
                            isinstance(_mv, _types.ModuleType)
                            and "bert_layers" in _mk
                            and hasattr(_mv, "BertEncoder")
                        ):
                            _BertEncoder = _mv.BertEncoder
                            break
                except ImportError:
                    pass

                if _BertEncoder is not None:
                    def _fixed_rebuild_alibi(self, size: int, device=None) -> None:
                        import math as _m

                        n_heads = self.num_attention_heads

                        def _get_slopes(n: int):
                            def _pow2(n):
                                s = 2 ** (-2 ** -(_m.log2(n) - 3))
                                return [s * s ** i for i in range(n)]
                            if _m.log2(n).is_integer():
                                return _pow2(n)
                            c = 2 ** _m.floor(_m.log2(n))
                            return _pow2(c) + _get_slopes(2 * c)[0::2][: n - c]

                        # device=None → ikut torch.device() context (meta saat init,
                        # cpu saat dipanggil manual setelah loading)
                        cp = torch.arange(size, device=device)[:, None]
                        mp = torch.arange(size, device=device)[None, :]
                        rp = torch.abs(mp - cp).unsqueeze(0).expand(n_heads, -1, -1)
                        # FIX: slopes pada device yang SAMA dengan cp (bukan selalu CPU)
                        slopes = torch.tensor(
                            _get_slopes(n_heads), dtype=torch.float32, device=cp.device
                        )
                        alibi = slopes.view(n_heads, 1, 1) * -rp
                        self._current_alibi_size = size
                        self.alibi = alibi.unsqueeze(0)

                    _BertEncoder.rebuild_alibi_tensor = _fixed_rebuild_alibi
                    print("[DNASequenceEncoder] BertEncoder.rebuild_alibi_tensor berhasil di-patch.")
                else:
                    print(
                        "[DNASequenceEncoder] PERINGATAN: patch BertEncoder tidak dilakukan "
                        "(kelas tidak ditemukan di sys.modules — kemungkinan versi "
                        "transformers tidak kompatibel). Error meta-device mungkin muncul."
                    )

                # ── FASE 2: Load model (menggunakan kelas ter-patch) ────────────
                self.encoder = AutoModel.from_pretrained(
                    backbone, config=config, trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    _fast_init=False,
                )

                # ── FASE 3: Materialize sisa meta tensor setelah from_pretrained ─
                # named_parameters() dan named_buffers() untuk parameter/buffer biasa.
                def _mat_cpu(m: "nn.Module") -> None:
                    for _nm, _p in list(m.named_parameters()):
                        if not _p.is_meta:
                            continue
                        _fill = torch.zeros(_p.shape, dtype=_p.dtype, device="cpu")
                        _sub = m
                        for _pt in _nm.split(".")[:-1]:
                            _sub = getattr(_sub, _pt)
                        _sub._parameters[_nm.split(".")[-1]] = nn.Parameter(
                            _fill, requires_grad=_p.requires_grad
                        )
                    for _nm, _b in list(m.named_buffers()):
                        if _b is None or not _b.is_meta:
                            continue
                        _fill = torch.zeros(_b.shape, dtype=_b.dtype, device="cpu")
                        _sub = m
                        for _pt in _nm.split(".")[:-1]:
                            _sub = getattr(_sub, _pt)
                        _sub._buffers[_nm.split(".")[-1]] = _fill

                _mat_cpu(self.encoder)

                # ── Null-out flash_attn_qkvpacked_func di semua modul bert_layers ──
                # DNABERT-2 bert_layers.py memilih Triton vs torch dengan memeriksa
                # apakah `flash_attn_qkvpacked_func is None` (baris ~161), BUKAN via
                # attn_impl config. Triton ≥2.1 menghapus `trans_b` dari tl.dot() →
                # CompilationError pada first forward pass.
                # Fix: set simbol ke None di semua modul bert_layers yang ada di
                # sys.modules agar forward() selalu masuk ke cabang PyTorch native.
                for _mk, _mv in list(_sys.modules.items()):
                    if "bert_layers" in _mk and isinstance(_mv, _types.ModuleType):
                        if getattr(_mv, "flash_attn_qkvpacked_func", None) is not None:
                            _mv.flash_attn_qkvpacked_func = None
                            print(f"[DNASequenceEncoder] flash_attn dinon-aktifkan di {_mk!r}.")

                # BertEncoder.alibi = plain tensor attribute (bukan buffer/parameter).
                # Setelah from_pretrained di dalam meta context, alibi masih meta.
                # Module.to() melewatinya → rebuild manual pada CPU.
                for _mod in self.encoder.modules():
                    if (
                        hasattr(_mod, "alibi")
                        and isinstance(_mod.alibi, torch.Tensor)
                        and _mod.alibi.is_meta
                        and hasattr(_mod, "rebuild_alibi_tensor")
                        and hasattr(_mod, "_current_alibi_size")
                    ):
                        _mod.rebuild_alibi_tensor(
                            size=_mod._current_alibi_size, device="cpu"
                        )

                if use_lora:
                    # DNABERT-2 (MosaicBERT) menggunakan fused QKV:
                    # satu Linear 'Wqkv' (768→2304) per attention layer,
                    # BUKAN modul 'query'/'key'/'value' terpisah seperti HF BERT.
                    lora_cfg = LoraConfig(
                        task_type=TaskType.FEATURE_EXTRACTION,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        target_modules=["Wqkv"],
                        bias="none",
                    )
                    self.encoder = get_peft_model(self.encoder, lora_cfg)

                    # Materialize PEFT-created meta tensors (LoRA A/B layers)
                    for _nm, _p in list(self.encoder.named_parameters()):
                        if not _p.is_meta:
                            continue
                        _fill = torch.empty(_p.shape, dtype=_p.dtype, device="cpu")
                        if ".lora_A." in _nm:
                            nn.init.kaiming_uniform_(_fill, a=_math.sqrt(5))
                        else:
                            nn.init.zeros_(_fill)
                        _sub = self.encoder
                        for _pt in _nm.split(".")[:-1]:
                            _sub = getattr(_sub, _pt)
                        _sub._parameters[_nm.split(".")[-1]] = nn.Parameter(
                            _fill, requires_grad=_p.requires_grad
                        )

                self._using_pretrained = True
                print(f"[DNASequenceEncoder] DNABERT-2 berhasil dimuat (mode='{encoder_mode}').")
                if use_lora:
                    self.encoder.print_trainable_parameters()

            except (ImportError, OSError) as e:
                raise RuntimeError(
                    f"[DNASequenceEncoder] Mode '{encoder_mode}' wajib menggunakan DNABERT-2, "
                    f"namun gagal dimuat: {e}. "
                    f"Gunakan encoder_mode='scratch' untuk melatih dari awal."
                ) from e

    def _build_scratch_encoder(
        self, vocab_size: int, hidden_dim: int, num_layers: int
    ) -> nn.Module:
        """Buat Transformer encoder minimal dari awal (tanpa pre-training)."""
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        class ScratchEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
                self.pos_enc   = nn.Embedding(512, hidden_dim)  # positional
                layer = TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
                    dropout=0.1, batch_first=True, norm_first=True,
                )
                self.transformer = TransformerEncoder(layer, num_layers=num_layers)

            def forward(self, input_ids, attention_mask=None, **kwargs):
                B, L = input_ids.shape
                pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
                x   = self.embedding(input_ids) + self.pos_enc(pos)
                # TransformerEncoder menggunakan src_key_padding_mask (True = ignore)
                if attention_mask is not None:
                    pad_mask = (attention_mask == 0)
                else:
                    pad_mask = None
                x = self.transformer(x, src_key_padding_mask=pad_mask)
                # Kembalikan sebagai namedtuple-like untuk kompatibilitas
                return type("EncoderOutput", (), {"last_hidden_state": x})()

        return ScratchEncoder()

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Aggregate token embeddings menjadi satu vektor sekuens.

        Strategi pooling:
          "mean" : rata-rata semua non-padding token (direkomendasikan untuk 16S)
          "cls"  : ambil token [CLS] saja (posisi 0)
          "max"  : max-pooling antar semua token
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)

        if self.pooling == "mean":
            sum_emb   = (hidden_states * mask).sum(dim=1)
            sum_mask  = mask.sum(dim=1).clamp(min=1e-9)
            return sum_emb / sum_mask

        if self.pooling == "max":
            hidden_states = hidden_states.masked_fill(mask == 0, float("-inf"))
            return hidden_states.max(dim=1).values

        raise ValueError(f"Pooling tidak dikenal: {self.pooling}")

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Parameter:
            input_ids      : (B, L) LongTensor
            attention_mask : (B, L) LongTensor

        Return:
            (B, hidden_dim) float sequence embeddings
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # DNABERT-2/MosaicBERT kadang mengembalikan tuple bukan BaseModelOutput.
        # Elemen pertama tuple adalah last_hidden_state.
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state          # (B, L, hidden_dim)
        elif isinstance(out, (tuple, list)):
            hidden = out[0]                         # (B, L, hidden_dim)
        else:
            raise TypeError(f"Output encoder tidak dikenal: {type(out)}")
        return self.pool(hidden, attention_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE FUSION MODULE
# ─────────────────────────────────────────────────────────────────────────────

class HyperbolicFusion(nn.Module):
    """
    Fusi embedding Euclidean (dari Transformer) dan Poincaré (dari HyTaxGNN).
    Menggunakan Möbius Addition sebagai pengganti concatenation di ruang hiperbolik.

    Alur:
      z_seq  (Euclidean, d=768) → linear projection → z_proj (Euclidean, d=128)
      z_proj → expmap0          → z_hyp  (Poincaré, d=128)
      z_query_node (Poincaré)    ← lookup node embedding terdekat untuk query
      z_fused = z_hyp ⊕_c z_query_node

    Parameter:
        seq_dim     : dimensi input embedding sekuens (Euclidean)
        hyp_dim     : dimensi Poincaré ball
        ball        : instance PoincareBall (shared dengan HyTaxGNN)

    Input:
        z_seq    : (B, seq_dim) Euclidean sequence embeddings
        node_emb : (N_nodes, hyp_dim) semua node embeddings di Poincaré ball

    Output:
        (B, hyp_dim) fused embeddings di Poincaré ball
    """

    def __init__(self, seq_dim: int, hyp_dim: int, ball: PoincareBall):
        super().__init__()
        self.ball    = ball
        self.proj    = nn.Linear(seq_dim, hyp_dim, bias=True)  # Euclidean projection
        self.dropout = nn.Dropout(0.1)

    def forward(self, z_seq: Tensor, node_emb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return:
            z_fused     : (B, hyp_dim) — fused embedding di Poincaré space
            z_hyp       : (B, hyp_dim) — query embedding di Poincaré space (sebelum fusion)
        """
        # Project ke hyp_dim di tangent space, lalu masuk ke Poincaré ball
        z_proj = self.dropout(self.proj(z_seq))  # (B, hyp_dim) Euclidean
        z_hyp  = self.ball.expmap0(z_proj)       # (B, hyp_dim) Poincaré

        # Cari node terdekat — gunakan tangent space L2 via torch.cdist
        # (menghindari OOM: pairwise_dist memperluas ke (B,N,d) ~2GB untuk N=61K)
        with torch.no_grad():
            z_tan  = self.ball.logmap0(z_hyp)                         # (B, d) Euclidean
            n_tan  = self.ball.logmap0(node_emb)                      # (N, d) Euclidean
            dists  = torch.cdist(z_tan, n_tan)                        # (B, N) — O(B×N) memory
            topk   = dists.topk(k=5, dim=-1, largest=False).indices  # (B, 5)
            near   = node_emb[topk].mean(dim=1)                      # (B, hyp_dim)

        # Möbius Addition: fusi query dengan nearest neighbor centroid
        z_fused = self.ball.mobius_add(z_hyp, near)  # (B, hyp_dim) di Poincaré

        return z_fused, z_hyp


# ─────────────────────────────────────────────────────────────────────────────
# 3. HIERARCHICAL CLASSIFICATION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalClassifierHead(nn.Module):
    """
    Kepala klasifikasi hierarkis: 6 classifier independen (satu per rank taksonomi).
    Input adalah embedding fusi yang di-logmap ke tangent space untuk linear layers.

    Parameter:
        hyp_dim    : dimensi Poincaré embedding (input dari fusion)
        num_classes: dict {"Phylum": N1, "Class": N2, ...} — jumlah kelas per rank
        ball       : instance PoincareBall
        dropout    : dropout rate

    Input:
        z_fused : (B, hyp_dim) — fused embedding di Poincaré space

    Output:
        list of len-6 tensors, masing-masing shape (B, num_classes[rank])
    """

    RANK_NAMES = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]

    def __init__(
        self,
        hyp_dim: int,
        num_classes: Dict[str, int],
        ball: PoincareBall,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ball    = ball
        self.dropout = nn.Dropout(dropout)

        # Buat classifier terpisah untuk setiap rank
        self.classifiers = nn.ModuleDict({
            rank: nn.Sequential(
                nn.Linear(hyp_dim, hyp_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hyp_dim // 2, num_classes.get(rank, 2)),
            )
            for rank in self.RANK_NAMES
        })

    def forward(self, z_fused: Tensor) -> List[Tensor]:
        """
        Parameter:
            z_fused : (B, hyp_dim) di Poincaré ball

        Return:
            list of 6 tensors shape (B, num_classes_rank) — logits per rank
        """
        # Poincaré → tangent space untuk operasi linear
        h = self.ball.logmap0(z_fused)   # (B, hyp_dim) Euclidean
        h = self.dropout(h)

        logits_per_rank = []
        for rank in self.RANK_NAMES:
            logits_per_rank.append(self.classifiers[rank](h))  # (B, C_rank)

        return logits_per_rank


# ─────────────────────────────────────────────────────────────────────────────
# 4. OOD DETECTION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class OODDetectionHead(nn.Module):
    """
    Kepala deteksi Out-of-Distribution (OOD) berbasis hyperbolic margin.

    OOD Score: min jarak hiperbolik dari query ke semua leaf-level node.
    Query dengan OOD score > threshold δ dianggap novel / OOD.

    Parameter:
        ball      : instance PoincareBall
        fpr_target: target False Positive Rate untuk kalibrasi threshold δ

    Penggunaan:
        ood_head = OODDetectionHead(ball, fpr_target=0.05)
        # Kalibrasi: jalankan sekali pada validation set
        ood_head.calibrate(indist_scores)
        # Inference
        scores, is_ood = ood_head(z_hyp, leaf_embeddings)
    """

    def __init__(self, ball: PoincareBall, fpr_target: float = 0.05):
        super().__init__()
        self.ball       = ball
        self.fpr_target = fpr_target
        self.threshold  = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def compute_ood_score(self, z_hyp: Tensor, leaf_emb: Tensor) -> Tensor:
        """
        Hitung OOD score per sampel = min jarak hiperbolik ke semua leaf nodes.

        Parameter:
            z_hyp    : (B, d) — query embeddings di Poincaré ball
            leaf_emb : (L, d) — species-level leaf node embeddings

        Return:
            (B,) — OOD scores (lebih besar = lebih OOD)
        """
        # Pairwise hyperbolic distances: (B, L)
        dists = self.ball.pairwise_dist(z_hyp, leaf_emb)
        # OOD score = minimum distance ke leaf terdekat
        return dists.min(dim=-1).values  # (B,)

    @torch.no_grad()
    def calibrate(self, indist_scores: Tensor):
        """
        Kalibrasi threshold δ dari distribusi OOD scores pada in-distribution samples.
        δ = persentil (1 - fpr_target) dari scores in-distribution.

        Parameter:
            indist_scores : (N,) OOD scores dari validation set in-distribution
        """
        percentile = (1.0 - self.fpr_target) * 100
        delta = torch.quantile(indist_scores, 1.0 - self.fpr_target)
        self.threshold.fill_(delta.item())
        print(f"[OODHead] Threshold dikalibrasi: δ = {delta.item():.4f} "
              f"(FPR target = {self.fpr_target*100:.0f}%)")

    def forward(
        self, z_hyp: Tensor, leaf_emb: Tensor, leaf_node_ids: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Parameter:
            z_hyp         : (B, d) query embeddings di Poincaré ball
            leaf_emb      : (L, d) leaf node embeddings
            leaf_node_ids : (L,) int — global IDs leaf nodes (untuk LCA lookup)

        Return:
            dict dengan:
              "ood_score"    : (B,) — OOD scores
              "is_ood"       : (B,) bool — apakah OOD
              "topk_leaf_ids": (B, 5) — top-5 nearest leaf node IDs
        """
        scores = self.compute_ood_score(z_hyp, leaf_emb)   # (B,)
        is_ood = scores > self.threshold

        # Top-5 nearest leaf nodes untuk LCA abstention
        dists  = self.ball.pairwise_dist(z_hyp, leaf_emb)  # (B, L)
        topk   = dists.topk(k=min(5, leaf_emb.shape[0]),
                            dim=-1, largest=False).indices  # (B, 5)

        if leaf_node_ids is not None:
            topk_ids = leaf_node_ids[topk]
        else:
            topk_ids = topk

        return {
            "ood_score":     scores,
            "is_ood":        is_ood,
            "topk_leaf_ids": topk_ids,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL: TaxoGraph-BERT
# ─────────────────────────────────────────────────────────────────────────────

class TaxoGraphBERT(nn.Module):
    """
    Model lengkap TaxoGraph-BERT.

    Alur forward:
      input_ids + attention_mask
          ↓ DNASequenceEncoder
      z_seq (B, 768)
          ↓ HyperbolicFusion ← node_emb dari HyTaxGNN
      z_fused (B, 128) di Poincaré ball
          ↓ HierarchicalClassifierHead
      logits[0..5] (B, C_rank) per rank
          ↓ OODDetectionHead ← leaf embeddings
      ood_score (B,), is_ood (B,)

    Parameter:
        cfg          : Config object
        num_classes  : dict {"Phylum": N1, ...} dari label encoder
        num_nodes    : total nodes dalam taxonomy graph
        edge_index   : (2, E) taxonomy tree edges — simpan sebagai buffer

    Penggunaan:
        model = TaxoGraphBERT(cfg, num_classes, num_nodes, edge_index)
        out   = model(input_ids, attention_mask)
        logits_genus = out["logits"][4]   # logits untuk rank Genus (index 4)
        ood_scores   = out["ood_score"]
    """

    def __init__(
        self,
        cfg,
        num_classes: Dict[str, int],
        num_nodes: int,
        edge_index: Tensor,
        leaf_node_ids: Optional[Tensor] = None,
    ):
        super().__init__()
        mcfg = cfg.model
        self.ball = PoincareBall(c=mcfg.poincare_curvature)

        # ── Komponen ──────────────────────────────────────────────────────
        # encoder_mode "all" tidak valid untuk single model instance — gunakan "pretrain"
        _effective_mode = mcfg.encoder_mode if mcfg.encoder_mode != "all" else "pretrain"
        self.seq_encoder = DNASequenceEncoder(
            backbone=mcfg.transformer_backbone,
            hidden_dim=mcfg.transformer_hidden_dim,
            pooling=mcfg.pooling_strategy,
            use_lora=mcfg.use_lora,
            lora_r=mcfg.lora_r,
            lora_alpha=mcfg.lora_alpha,
            lora_dropout=mcfg.lora_dropout,
            encoder_mode=_effective_mode,
        )

        self.hytaxgnn = HyTaxGNN(
            num_nodes=num_nodes,
            in_dim=64,
            hidden_dim=mcfg.poincare_dim,
            n_layers=mcfg.hgnn_num_layers,
            curvature=mcfg.poincare_curvature,
            dropout=mcfg.hgnn_dropout,
        )

        self.fusion = HyperbolicFusion(
            seq_dim=mcfg.transformer_hidden_dim,
            hyp_dim=mcfg.poincare_dim,
            ball=self.ball,
        )

        self.classifier = HierarchicalClassifierHead(
            hyp_dim=mcfg.poincare_dim,
            num_classes=num_classes,
            ball=self.ball,
            dropout=mcfg.classifier_dropout,
        )

        self.ood_head = OODDetectionHead(
            ball=self.ball,
            fpr_target=mcfg.ood_fpr_target,
        )

        # Simpan edge_index sebagai buffer (bukan parameter) — tidak diupdate
        self.register_buffer("edge_index", edge_index)

        # Simpan leaf_node_ids (Species-level) sebagai buffer untuk OOD detection
        self.register_buffer("leaf_node_ids", leaf_node_ids)

        # Cache node embeddings (di-update setiap epoch Phase 1, frozen setelahnya)
        self._node_emb_cache: Optional[Tensor] = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_node_embeddings(self) -> Tensor:
        """
        Jalankan HyTaxGNN untuk mendapatkan semua node embeddings.
        Di-cache setelah Phase 1 pre-training selesai.
        """
        return self.hytaxgnn(self.edge_index)  # (N_nodes, poincare_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        update_graph: bool = True,
    ) -> Dict[str, object]:
        """
        Forward pass lengkap.

        Parameter:
            input_ids      : (B, L) — token IDs
            attention_mask : (B, L) — attention mask
            update_graph   : apakah hitung ulang node embeddings (True saat training,
                             False saat inference untuk efisiensi)

        Return:
            dict dengan:
              "logits"     : list of 6 tensors (B, C_rank) — logits per rank
              "ood_score"  : (B,) — OOD score
              "is_ood"     : (B,) bool
              "hyp_emb"    : (B, poincare_dim) — embedding query di Poincaré space
              "topk_leaf_ids": (B, 5) — leaf node IDs terdekat
        """
        # ── 1. Sequence encoder ──────────────────────────────────────────
        z_seq = self.seq_encoder(input_ids, attention_mask)  # (B, 768)

        # ── 2. Taxonomy graph embedding ──────────────────────────────────
        if update_graph or self._node_emb_cache is None:
            node_emb = self.get_node_embeddings()            # (N_nodes, 128)
            self._node_emb_cache = node_emb.detach()
        else:
            node_emb = self._node_emb_cache

        # ── 3. Feature fusion ────────────────────────────────────────────
        z_fused, z_hyp = self.fusion(z_seq, node_emb)       # (B, 128)

        # ── 4. Hierarchical classification ───────────────────────────────
        logits = self.classifier(z_fused)                   # list of 6 (B, C)

        # ── 5. OOD detection ─────────────────────────────────────────────
        # Gunakan self.leaf_node_ids (Species-level buffer) untuk OOD scoring
        leaf_emb = node_emb[self.leaf_node_ids]  # (L, d) — hanya Species nodes
        ood_out  = self.ood_head(z_hyp, leaf_emb, leaf_node_ids=self.leaf_node_ids)

        return {
            "logits":        logits,
            "ood_score":     ood_out["ood_score"],
            "is_ood":        ood_out["is_ood"],
            "hyp_emb":       z_hyp,           # renamed dari z_hyp → hyp_emb
            "topk_leaf_ids": ood_out["topk_leaf_ids"],
        }


    @torch.no_grad()
    def predict_with_abstention(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        conf_thresholds: Optional[Dict[str, float]] = None,
        node2idx: Optional[dict] = None,
        label_encoder=None,
    ) -> List[Dict]:
        """
        Top-down prediction dengan abstention berbasis confidence threshold.
        Model berhenti di rank pertama yang confidence-nya < threshold τ_r;
        jika OOD terdeteksi, gunakan compute_lca untuk LCA abstention.

        Parameter:
            input_ids       : (B, L) LongTensor
            attention_mask  : (B, L) LongTensor
            conf_thresholds : dict {rank_name: float} — threshold per rank
                              Default: {"Phylum": 0.5, "Class": 0.4, ..., "Species": 0.3}
            node2idx        : dict dari build_taxonomy_graph() (untuk LCA)
            label_encoder   : HierarchicalLabelEncoder (untuk LCA + decode)

        Return:
            list of dicts per sample:
              {"predictions": {rank: label}, "abstain_rank": rank|None, "is_ood": bool,
               "ood_score": float, "lca_rank": rank|None}
        """
        from hyperbolic import compute_lca

        RANK_NAMES = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]
        default_thresholds = {
            "Phylum": 0.5, "Class": 0.45, "Order": 0.4,
            "Family": 0.35, "Genus": 0.3, "Species": 0.25,
        }
        tau = conf_thresholds or default_thresholds

        was_training = self.training
        self.eval()

        out      = self.forward(input_ids, attention_mask, update_graph=False)
        logits   = out["logits"]           # list of 6 (B, C)
        ood_scores = out["ood_score"]      # (B,)
        is_ood   = out["is_ood"]           # (B,)
        topk_ids = out["topk_leaf_ids"]    # (B, k)

        B = input_ids.shape[0]
        results = []

        for i in range(B):
            predictions = {}
            abstain_rank = None
            lca_rank = None

            if is_ood[i].item() and node2idx is not None and label_encoder is not None:
                # OOD: hitung LCA dari top-k leaf nodes
                lca_out  = compute_lca(topk_ids[i], node2idx, label_encoder)
                predictions = lca_out["predictions"]
                lca_rank    = lca_out["lca_rank"]
            else:
                # In-distribution: top-down prediction dengan threshold
                for r, rank in enumerate(RANK_NAMES):
                    probs_r = torch.softmax(logits[r][i], dim=-1)  # (C,)
                    conf_r, pred_r = probs_r.max(dim=-1)

                    if conf_r.item() < tau.get(rank, 0.3):
                        abstain_rank = rank
                        break  # berhenti di rank ini

                    # Decode prediction ke label string
                    if label_encoder is not None:
                        label = label_encoder.decode(rank, pred_r.item())
                    else:
                        label = str(pred_r.item())
                    predictions[rank] = label

            results.append({
                "predictions":  predictions,
                "abstain_rank": abstain_rank,
                "is_ood":       is_ood[i].item(),
                "ood_score":    ood_scores[i].item(),
                "lca_rank":     lca_rank,
            })

        if was_training:
            self.train()

        return results


# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _materialize_meta_tensors(module: nn.Module) -> int:
    """
    Scan every parameter and buffer in *module* recursively using full dotted
    names (named_parameters / named_buffers). Any tensor still on the PyTorch
    'meta' device is replaced with a real CPU tensor so that model.to(device)
    does not raise:
        RuntimeError: Tensor on device meta is not on the expected device cpu!

    Uses accelerate.utils.set_module_tensor_to_device when available (handles
    tied weights and hooks correctly), otherwise falls back to manual patching
    via the dotted attribute path.
    LoRA A matrices get kaiming-uniform init; everything else is zeros.
    Returns the number of tensors that were materialised.
    """
    import math as _math

    try:
        from accelerate.utils import set_module_tensor_to_device as _set
        _use_acc = True
    except ImportError:
        _use_acc = False

    count = 0

    # ── parameters ──────────────────────────────────────────────────────────
    for param_name, param in list(module.named_parameters()):
        if not param.is_meta:
            continue
        fill = torch.empty(param.shape, dtype=param.dtype, device="cpu")
        if ".lora_A." in param_name or param_name.endswith(".lora_A"):
            nn.init.kaiming_uniform_(fill, a=_math.sqrt(5))
        else:
            nn.init.zeros_(fill)
        if _use_acc:
            _set(module, param_name, "cpu", value=fill)
        else:
            parts = param_name.split(".")
            m = module
            for part in parts[:-1]:
                m = getattr(m, part)
            m._parameters[parts[-1]] = nn.Parameter(
                fill, requires_grad=param.requires_grad
            )
        count += 1

    # ── buffers ─────────────────────────────────────────────────────────────
    for buf_name, buf in list(module.named_buffers()):
        if buf is None or not buf.is_meta:
            continue
        fill = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
        if _use_acc:
            _set(module, buf_name, "cpu", value=fill)
        else:
            parts = buf_name.split(".")
            m = module
            for part in parts[:-1]:
                m = getattr(m, part)
            m._buffers[parts[-1]] = fill
        count += 1

    return count


# ───────────────────────────────────────────────────────────────────────────────
# BUILDER
# ───────────────────────────────────────────────────────────────────────────────

def build_model(cfg, label_encoder, edge_index: Tensor, leaf_node_ids: Optional[Tensor] = None) -> TaxoGraphBERT:
    """
    Bangun model TaxoGraphBERT dari config dan label encoder.

    Parameter:
        cfg           : Config object dari config.py
        label_encoder : HierarchicalLabelEncoder yang sudah di-fit
        edge_index    : (2, E) taxonomy tree edges

    Return:
        TaxoGraphBERT model siap training
    """
    num_classes = label_encoder.num_classes()
    num_nodes   = edge_index.max().item() + 1

    model = TaxoGraphBERT(
        cfg=cfg,
        num_classes=num_classes,
        num_nodes=num_nodes,
        edge_index=edge_index,
        leaf_node_ids=leaf_node_ids,
    )

    # Materialize any meta-device tensors before moving to the target device.
    # Catches meta tensors from DNABERT-2 custom code, PEFT init_empty_weights,
    # accelerate, etc. — regardless of origin.
    n_meta = _materialize_meta_tensors(model)
    if n_meta:
        print(f"[build_model] Materialized {n_meta} meta tensor(s) to CPU.")

    # Dukung MPS (Apple Silicon), CUDA, dan CPU langsung dari cfg.device
    device = torch.device(cfg.device)
    model  = model.to(device)

    # Auto-detect multi-GPU (DataParallel) — khusus CUDA
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        print(f"[build_model] Terdeteksi {n_gpus} GPU → menggunakan nn.DataParallel "
              f"(GPU: {', '.join(f'cuda:{i}' for i in range(n_gpus))})")
        model = torch.nn.DataParallel(model)
    elif n_gpus == 1:
        print(f"[build_model] Single GPU terdeteksi (cuda:0)")

    _base = model.module if isinstance(model, torch.nn.DataParallel) else model
    n_params = sum(p.numel() for p in _base.parameters())
    n_train  = sum(p.numel() for p in _base.parameters() if p.requires_grad)
    print(f"[build_model] Total params: {n_params:,} | Trainable: {n_train:,} "
          f"({n_train/n_params*100:.1f}%)")

    return model