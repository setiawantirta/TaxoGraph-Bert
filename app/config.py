"""
config.py — TaxoGraph-BERT Central Configuration
=================================================
Semua hyperparameter, path, dan flag operasional dikumpulkan di sini.
Ubah nilai di bagian ini sebelum menjalankan pipeline.

Penggunaan:
    from config import CFG
    model = TaxoGraphBERT(CFG)
    trainer = Trainer(CFG)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — sesuaikan ke lokasi data Anda
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PathConfig:
    # Input
    silva_fasta: str = "data/raw/SILVA_138_SSURef_tax_silva.fasta"
    silva_taxonomy: str = "data/raw/taxonomy_7_levels.txt"
    mockrobiota_dir: str = "data/external/mockrobiota/"
    ncbi_holdout_fasta: str = "data/external/ncbi_temporal_holdout.fasta"

    # Processed (auto-generated)
    hdf5_train: str = "data/processed/silva_train.h5"
    hdf5_val: str = "data/processed/silva_val.h5"
    taxonomy_graph: str = "data/processed/taxonomy_graph.pkl"
    label_encoder: str = "data/processed/label_encoder.pkl"
    vocab_bpe: str = "data/processed/bpe_vocab.json"

    # Outputs
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    plot_dir: str = "outputs/plots"
    metric_dir: str = "outputs/metrics"
    embedding_dir: str = "outputs/embeddings"


# ─────────────────────────────────────────────────────────────────────────────
# DATA — preprocessing & tokenization
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    # V3-V4 primer pair
    primer_fwd: str = "CCTACGGGNGGCWGCAG"      # 341F
    primer_rev: str = "GACTACHVGGGTATCTAATCC"   # 805R
    primer_max_mismatch: int = 2                 # maks mismatch per primer

    # Quality filtering
    max_ambiguous_bases: int = 5                 # sekuens > N basa ambigu → dibuang
    min_amplicon_len: int = 350                  # panjang minimum amplikon V3-V4 (bp)
    max_amplicon_len: int = 550                  # panjang maksimum amplikon V3-V4 (bp)

    # Tokenization
    kmer_k: int = 6                              # k untuk k-mer overlapping (k=6 optimal 16S)
    use_bpe: bool = True                         # aktifkan BPE fallback
    bpe_vocab_size: int = 4096                   # = 4^6 (setara k=6 coverage)
    max_seq_len: int = 512                       # panjang token maksimum input model

    # Large data
    # 4^6 = 4096 fitur k-mer, ~350,000 baris → gunakan HDF5 + chunked loading
    hdf5_chunk_size: int = 1024                  # ukuran chunk HDF5 (baris per chunk)
    dataloader_num_workers: int = 4              # parallel workers untuk DataLoader
    pin_memory: bool = True                      # pin memory GPU (True jika CUDA tersedia)
    prefetch_factor: int = 2                     # prefetch batches per worker

    # Taxonomic Roll-up
    min_samples_per_taxon: int = 5               # taksa < N → roll-up ke rank lebih tinggi
    rollup_max_rank: str = "Class"               # batas atas roll-up (tidak melampaui Class)

    # Train/Val split (internal, hanya untuk early stopping)
    val_fraction: float = 0.10                   # 10% val dari SILVA (stratified)
    random_seed: int = 42

    # Uninformative label keywords (akan dibersihkan)
    uninformative_keywords: List[str] = field(default_factory=lambda: [
        "uncultured", "unidentified", "metagenome",
        "environmental sample", "human gut", "bacterium",
        "unclassified", "unknown",
    ])

    # Taxonomic rank order (coarse → fine)
    tax_ranks: List[str] = field(default_factory=lambda: [
        "Phylum", "Class", "Order", "Family", "Genus", "Species"
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — arsitektur TaxoGraph-BERT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Transformer sequence encoder
    transformer_backbone: str = "zhihan1996/DNABERT-2-117M"  # HuggingFace model ID
    transformer_hidden_dim: int = 768            # dimensi hidden DNABERT-2
    pooling_strategy: str = "mean"               # "mean" | "cls" | "max"

    # LoRA (Parameter-Efficient Fine-Tuning)
    use_lora: bool = True
    lora_r: int = 16                             # rank LoRA
    lora_alpha: int = 32                         # scaling factor α
    lora_dropout: float = 0.05
    # DNABERT-2 (MosaicBERT) uses a single fused Linear 'Wqkv' (hidden→3*hidden)
    # instead of separate 'query'/'key'/'value' modules like standard HF BERT.
    lora_target_modules: List[str] = field(default_factory=lambda: ["Wqkv"])

    # Hyperbolic Taxonomy GNN (HyTaxGNN)
    hgnn_hidden_dim: int = 128                   # dim embedding hiperbolik setiap node
    hgnn_num_layers: int = 3                     # jumlah layer HGCN
    hgnn_dropout: float = 0.1
    poincare_curvature: float = 1.0              # kelengkungan c Poincaré ball (c=1 standar)
    poincare_dim: int = 128                      # dimensi Poincaré embedding

    # Feature fusion
    fusion_method: str = "mobius_add"            # "mobius_add" | "concat" | "attention"
    fusion_output_dim: int = 256                 # dimensi setelah fusion

    # Hierarchical classification head
    num_tax_ranks: int = 6                       # Phylum → Species
    classifier_dropout: float = 0.1

    # OOD detection head
    ood_method: str = "hyperbolic_margin"        # "hyperbolic_margin" | "energy" | "maxsoftmax"
    ood_fpr_target: float = 0.05                 # FPR target untuk kalibrasi threshold δ
    ood_topk_lca: int = 5                        # top-k leaf nodes untuk LCA abstention

    # Encoder mode
    encoder_mode: str = "pretrain"               # "pretrain" (default) | "scratch" | "all"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING — optimiser, scheduler, regularisasi
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Batch & accumulation
    batch_size: int = 32                         # per-GPU batch size
    gradient_accumulation_steps: int = 8        # effective batch = 32 × 8 = 256
    max_epochs: int = 100
    early_stopping_patience: int = 10           # stop jika val F1 tidak naik N epoch

    # Optimizer — Transformer backbone
    lr_transformer: float = 3e-5
    weight_decay_transformer: float = 0.01

    # Optimizer — HyTaxGNN (Riemannian SGD)
    lr_hgnn: float = 1e-4
    lr_classifier: float = 5e-4
    lr_ood_head: float = 5e-4

    # Scheduler
    scheduler_type: str = "cosine_with_warmup"   # "linear" | "cosine_with_warmup"
    warmup_ratio: float = 0.06                   # 6% total steps sebagai warmup

    # Loss
    hierarchical_penalty_gamma: float = 0.1     # γ: skala penalti lintas rank
    rank_weights: List[float] = field(default_factory=lambda: [
        0.5, 0.6, 0.7, 0.8, 1.0, 1.2           # Phylum → Species (makin halus makin besar)
    ])

    # Regularisasi
    label_smoothing: float = 0.05               # mencegah overconfidence
    max_grad_norm: float = 1.0                  # gradient clipping

    # Mixed precision
    use_amp: bool = True                        # Automatic Mixed Precision (FP16)

    # Phase training
    pretrain_hgnn_epochs: int = 200             # Phase 1: Poincaré pre-training HyTaxGNN
    pretrain_hgnn_lr: float = 0.01             # Riemannian SGD lr untuk Phase 1
    phase1_edge_batch_size: int = 4096         # edge per step Phase 1; kurangi jika VRAM <8 GB


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — metrik, backup, visualisasi
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    # OOD temporal hold-out
    ood_n_samples: int = 50                     # jumlah sekuens NCBI post-2021
    ood_indist_sample_n: int = 50               # jumlah sampel in-dist untuk perbandingan

    # Metrics
    compute_calibration: bool = True            # reliability diagram & ECE
    n_calibration_bins: int = 15               # bins untuk calibration diagram

    # Statistical tests
    run_mcnemar: bool = True                    # McNemar's Test antar model
    run_delong: bool = True                     # DeLong's Test untuk AUROC CI
    alpha: float = 0.05                         # significance level

    # Backup
    save_csv: bool = True                       # simpan tabel metrik ke CSV
    save_plots: bool = True                     # simpan semua plot ke PNG
    plot_dpi: int = 300                         # resolusi PNG
    plot_format: str = "png"                    # "png" | "pdf" | "svg"

    # Embedding visualisasi
    save_embeddings: bool = True                # simpan embeddings ke HDF5
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CONFIG — gabungkan semua
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Global
    device: str = "cuda"           # "cuda" | "cpu" | "mps" (Apple Silicon)
    seed: int = 42
    log_level: str = "INFO"
    experiment_name: str = "taxograph_bert_v1"


# Singleton instance — impor ini di semua modul
CFG = Config()