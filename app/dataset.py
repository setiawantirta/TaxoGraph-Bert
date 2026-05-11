"""
data/dataset.py — HDF5 Lazy-Loading Dataset untuk Data Besar
=============================================================
Strategi untuk menangani data besar (4^6 = 4096 fitur, ~350.000 baris):

  1. HDF5 Lazy Loading  : hanya membaca slice yang diminta dari disk,
                          tidak memuat semua data ke RAM sekaligus.
  2. Chunked Indexing   : dataset HDF5 dibuat dengan chunk_size sehingga
                          akses acak (random access) bersifat cache-friendly.
  3. Memmap Fallback    : jika HDF5 tidak tersedia, gunakan np.memmap sebagai
                          fallback untuk dataset sangat besar.
  4. DataLoader Config  : num_workers, pin_memory, prefetch_factor diatur via
                          CFG.data untuk paralelisasi I/O.
  5. k-mer Tokenization : konversi sekuens DNA → tensor integer on-the-fly
                          (tidak perlu pre-tokenize semua sekuens).

Penggunaan:
    from data.dataset import SILVADataset, build_dataloaders
    train_dl, val_dl = build_dataloaders(CFG)
    for batch in train_dl:
        seq_ids, input_ids, attention_mask, labels = batch
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# K-MER TOKENIZER (in-process, tanpa HuggingFace untuk efisiensi)
# ─────────────────────────────────────────────────────────────────────────────

class KmerTokenizer:
    """
    Tokenizer k-mer overlapping untuk sekuens DNA.
    Mengkonversi string DNA → list integer token ID.

    Vocab:
        0   : [PAD]
        1   : [UNK]  (basa ambigu N)
        2   : [CLS]
        3   : [SEP]
        4+  : setiap k-mer unik (A^k = 4096 untuk k=6)

    Parameter:
        k          : panjang k-mer (default 6)
        vocab_size : ukuran vocab maksimum (4^k + 4 token spesial)
        max_len    : panjang sequence maksimum (dalam token, bukan basa)

    Penggunaan:
        tok = KmerTokenizer(k=6, max_len=512)
        enc = tok.encode("ATCGATCGATCG")
        # enc = {"input_ids": tensor([2, 45, 89, ...]), "attention_mask": tensor([1, 1, 1, ...])}
    """

    BASES = "ATCG"
    PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3
    SPECIAL_OFFSET = 4

    def __init__(self, k: int = 6, max_len: int = 512):
        self.k = k
        self.max_len = max_len

        # Buat vocab lengkap semua k-mer (4^k)
        self._build_vocab()

    def _build_vocab(self):
        """Buat mapping kmer_string → integer ID."""
        import itertools
        self.kmer2id: Dict[str, int] = {}
        for kmer in itertools.product(self.BASES, repeat=self.k):
            kmer_str = "".join(kmer)
            self.kmer2id[kmer_str] = len(self.kmer2id) + self.SPECIAL_OFFSET
        self.vocab_size = len(self.kmer2id) + self.SPECIAL_OFFSET  # 4096 + 4 = 4100

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Encode satu sekuens DNA menjadi input_ids + attention_mask.

        Parameter:
            sequence : string DNA (hanya A/T/C/G/N)

        Return:
            dict dengan key "input_ids" (LongTensor) dan "attention_mask" (LongTensor)
            keduanya shape (max_len,)
        """
        seq = sequence.upper()
        # Sliding window k-mer
        tokens = [self.CLS_ID]
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k]
            tokens.append(self.kmer2id.get(kmer, self.UNK_ID))
        tokens.append(self.SEP_ID)

        # Truncate
        tokens = tokens[:self.max_len]

        # Pad
        attn = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
        tokens = tokens + [self.PAD_ID] * (self.max_len - len(tokens))

        return {
            "input_ids":      torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attn,   dtype=torch.long),
        }

    def batch_encode(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode sekumpulan sekuens sekaligus (lebih efisien untuk batch).

        Parameter:
            sequences : list string DNA

        Return:
            dict dengan "input_ids" dan "attention_mask" shape (B, max_len)
        """
        encoded = [self.encode(s) for s in sequences]
        return {
            "input_ids":      torch.stack([e["input_ids"]      for e in encoded]),
            "attention_mask": torch.stack([e["attention_mask"] for e in encoded]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BPE TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    BPE (Byte-Pair Encoding) tokenizer untuk sekuens DNA.
    Dilatih dari corpus DNA menggunakan library HuggingFace `tokenizers`.
    Jika `tokenizers` tidak tersedia, fallback ke KmerTokenizer.

    Parameter:
        vocab_size : ukuran vocab BPE (default 4096 = setara k=6)
        max_len    : panjang sequence maksimum (dalam token)
        vocab_path : path simpan/muat vocab JSON (opsional)

    Penggunaan:
        tok = BPETokenizer(vocab_size=4096, max_len=512)
        tok.train(sequences)              # atau
        tok = BPETokenizer.load("bpe_vocab.json")
        enc = tok.encode("ATCGATCGATCG")
    """

    PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3

    def __init__(self, vocab_size: int = 4096, max_len: int = 512):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self._tokenizer = None
        self._available = self._check_library()

    @staticmethod
    def _check_library() -> bool:
        try:
            import tokenizers  # noqa: F401
            return True
        except ImportError:
            return False

    def train(self, sequences: List[str], vocab_path: Optional[str] = None) -> "BPETokenizer":
        """
        Latih BPE tokenizer pada korpus sekuens DNA.

        Parameter:
            sequences  : list string DNA
            vocab_path : simpan vocab ke file JSON ini jika diberikan

        Return:
            self
        """
        if not self._available:
            from loguru import logger
            logger.warning("Library `tokenizers` tidak tersedia. BPETokenizer tidak dilatih.")
            return self

        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import CharDelimiterSplit

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = CharDelimiterSplit(" ")

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            min_frequency=2,
            show_progress=True,
        )
        tokenizer.train_from_iterator(sequences, trainer=trainer)
        self._tokenizer = tokenizer

        if vocab_path:
            Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(vocab_path))

        return self

    @classmethod
    def load(cls, vocab_path: str, max_len: int = 512) -> "BPETokenizer":
        """Muat BPE tokenizer dari file JSON."""
        obj = cls(max_len=max_len)
        if not obj._available:
            return obj
        from tokenizers import Tokenizer
        obj._tokenizer = Tokenizer.from_file(str(vocab_path))
        return obj

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Encode satu sekuens DNA menjadi input_ids + attention_mask.
        Output shape identik dengan KmerTokenizer.encode().
        """
        if self._tokenizer is None:
            raise RuntimeError("BPETokenizer belum dilatih. Panggil .train() atau .load() dulu.")

        enc = self._tokenizer.encode(sequence.upper())
        tokens = [self.CLS_ID] + enc.ids[:self.max_len - 2] + [self.SEP_ID]
        attn   = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
        tokens = tokens + [self.PAD_ID] * (self.max_len - len(tokens))

        return {
            "input_ids":      torch.tensor(tokens[:self.max_len], dtype=torch.long),
            "attention_mask": torch.tensor(attn[:self.max_len],   dtype=torch.long),
        }

    @property
    def is_trained(self) -> bool:
        return self._tokenizer is not None


# ─────────────────────────────────────────────────────────────────────────────
# DUAL TOKENIZER (k-mer + BPE wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class DualTokenizer:
    """
    Wrapper yang menggabungkan KmerTokenizer dan BPETokenizer.
    Jika use_bpe=True dan BPETokenizer sudah dilatih, gunakan BPE;
    jika tidak tersedia atau belum dilatih, fallback ke k-mer tanpa error.

    Output dict identik dengan KmerTokenizer.encode() sehingga Dataset
    tidak perlu diubah.

    Parameter:
        kmer_tok : instance KmerTokenizer
        bpe_tok  : instance BPETokenizer (boleh None)
        use_bpe  : apakah prefer BPE jika tersedia

    Penggunaan:
        tok = DualTokenizer(kmer_tok, bpe_tok, use_bpe=True)
        enc = tok.encode("ATCGATCG")  # gunakan BPE jika tersedia
    """

    def __init__(
        self,
        kmer_tok: KmerTokenizer,
        bpe_tok: Optional["BPETokenizer"] = None,
        use_bpe: bool = True,
    ):
        self.kmer_tok = kmer_tok
        self.bpe_tok  = bpe_tok
        self.use_bpe  = use_bpe
        self._using_bpe = (
            use_bpe
            and bpe_tok is not None
            and bpe_tok.is_trained
        )

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Encode sekuens — gunakan BPE jika tersedia, else k-mer."""
        if self._using_bpe:
            return self.bpe_tok.encode(sequence)
        return self.kmer_tok.encode(sequence)

    def batch_encode(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Batch encode."""
        encoded = [self.encode(s) for s in sequences]
        return {
            "input_ids":      torch.stack([e["input_ids"]      for e in encoded]),
            "attention_mask": torch.stack([e["attention_mask"] for e in encoded]),
        }

    @property
    def vocab_size(self) -> int:
        if self._using_bpe:
            return self.bpe_tok.vocab_size
        return self.kmer_tok.vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 LAZY-LOADING DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SILVADataset(Dataset):
    """
    Dataset PyTorch dengan lazy-loading dari HDF5.
    File HDF5 TIDAK dibuka seluruhnya ke RAM; setiap item dibaca on-demand.

    Strategi data besar:
      - File HDF5 tetap di disk; hanya chunk yang diperlukan dibaca.
      - Setiap worker DataLoader membuka file HDF5 sendiri (tidak di __init__)
        karena h5py file handle tidak bisa di-share antar proses.
      - Tokenisasi dilakukan on-the-fly (bukan pre-computed) untuk menghemat RAM.

    Parameter:
        hdf5_path  : path ke file HDF5 hasil preprocess.py
        tokenizer  : instance KmerTokenizer
        augment    : apakah terapkan augmentasi sekuens (reverse complement)
        cache_size : jumlah item yang di-cache di RAM (0 = tanpa cache)

    Penggunaan:
        ds = SILVADataset("data/processed/silva_train.h5", tokenizer)
        item = ds[0]  # dict dengan seq_id, input_ids, attention_mask, labels
    """

    def __init__(
        self,
        hdf5_path: str,
        tokenizer: KmerTokenizer,
        augment: bool = False,
        cache_size: int = 0,
    ):
        self.hdf5_path = hdf5_path
        self.tokenizer = tokenizer
        self.augment = augment
        self.cache_size = cache_size
        self._cache: Dict[int, dict] = {}

        # Buka sementara hanya untuk mendapatkan metadata
        with h5py.File(hdf5_path, "r") as f:
            self.n_samples = f.attrs["n_samples"]
            self.n_ranks   = f.attrs["n_ranks"]

        # File handle akan dibuka di worker setelah fork (lihat _get_handle)
        self._file_handle = None

    def __len__(self) -> int:
        return self.n_samples

    def _get_handle(self) -> h5py.File:
        """
        Buka file HDF5 lazily per-worker.
        Penting: h5py file tidak thread-safe; setiap worker membuka sendiri.
        """
        if self._file_handle is None:
            self._file_handle = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._file_handle

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Cek cache
        if idx in self._cache:
            return self._cache[idx]

        f = self._get_handle()
        seq    = f["sequences"][idx]
        label  = f["labels"][idx]          # shape (6,) int32
        seq_id = f["seq_ids"][idx]

        # Decode bytes jika perlu
        if isinstance(seq, bytes):
            seq = seq.decode("utf-8")
        if isinstance(seq_id, bytes):
            seq_id = seq_id.decode("utf-8")

        # Augmentasi: 50% chance reverse complement
        if self.augment and torch.rand(1).item() > 0.5:
            seq = self._reverse_complement(seq)

        # Tokenisasi on-the-fly
        encoded = self.tokenizer.encode(seq)

        item = {
            "seq_id":         seq_id,
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels":         torch.tensor(label, dtype=torch.long),
        }

        # Cache jika diminta (berguna untuk dataset kecil)
        if self.cache_size > 0 and len(self._cache) < self.cache_size:
            self._cache[idx] = item

        return item

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Hitung reverse complement sekuens DNA."""
        comp = str.maketrans("ATCGN", "TAGCN")
        return seq.translate(comp)[::-1]

    def __del__(self):
        """Tutup file handle saat objek dihancurkan."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER — atasi imbalance saat training
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(dataset: SILVADataset, rank_idx: int = 4) -> WeightedRandomSampler:
    """
    Buat WeightedRandomSampler berdasarkan frekuensi kelas pada rank tertentu.
    Sampel dari kelas minoritas akan lebih sering muncul dalam epoch.

    Parameter:
        dataset   : SILVADataset instance
        rank_idx  : index rank untuk bobot (0=Phylum, ..., 4=Genus, 5=Species)

    Return:
        WeightedRandomSampler siap dipakai di DataLoader
    """
    from loguru import logger

    logger.info(f"Menghitung class weights untuk rank idx={rank_idx}...")
    f = dataset._get_handle()
    labels = f["labels"][:, rank_idx]  # baca semua label rank tertentu sekaligus

    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)  # inverse frequency

    sample_weights = torch.tensor(
        [class_weights[l] for l in tqdm(labels, desc="Sample weights", miniters=10000)],
        dtype=torch.float,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.info(f"WeightedRandomSampler dibuat: {len(sample_weights):,} sampel")
    return sampler


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate untuk DataLoader.
    seq_id adalah string, jadi tidak bisa distack sebagai tensor biasa.
    """
    seq_ids        = [item["seq_id"] for item in batch]
    input_ids      = torch.stack([item["input_ids"]      for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels         = torch.stack([item["labels"]         for item in batch])
    return {
        "seq_ids":        seq_ids,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BUILDER — entry point utama untuk modul lain
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Bangun train dan val DataLoader dari config.

    Strategi data besar yang diterapkan:
      - HDF5 lazy loading (SILVADataset)
      - WeightedRandomSampler untuk train (atasi imbalance)
      - num_workers > 0 untuk parallel I/O
      - pin_memory=True untuk transfer CPU→GPU yang cepat
      - prefetch_factor=2 untuk pipeline loading

    Parameter:
        cfg : Config object dari config.py

    Return:
        (train_dataloader, val_dataloader)
    """
    from loguru import logger

    kmer_tok = KmerTokenizer(
        k=cfg.data.kmer_k,
        max_len=cfg.data.max_seq_len,
    )

    # Buat DualTokenizer — gunakan BPE jika tersedia dan diaktifkan
    bpe_tok = None
    if getattr(cfg.data, "use_bpe", False):
        vocab_path = getattr(cfg.paths, "vocab_bpe", None)
        if vocab_path and Path(vocab_path).exists():
            try:
                bpe_tok = BPETokenizer.load(vocab_path, max_len=cfg.data.max_seq_len)
                logger.info(f"BPE tokenizer dimuat dari {vocab_path}")
            except Exception as e:
                logger.warning(f"Gagal muat BPE vocab: {e}. Fallback ke k-mer.")
        else:
            logger.info("BPE vocab belum ada; gunakan k-mer tokenizer.")

    tokenizer = DualTokenizer(kmer_tok, bpe_tok, use_bpe=getattr(cfg.data, "use_bpe", False))
    logger.info(f"Tokenizer aktif: {'BPE' if tokenizer._using_bpe else 'k-mer (k=%d)' % cfg.data.kmer_k}")

    # Cek apakah HDF5 sudah tersedia
    train_path = cfg.paths.hdf5_train
    val_path   = cfg.paths.hdf5_val

    if not Path(train_path).exists():
        logger.warning(f"HDF5 train tidak ditemukan di {train_path}. "
                       f"Jalankan data/preprocess.py terlebih dahulu.")
        # Return dummy loader untuk testing
        return _build_dummy_loaders(cfg, tokenizer)

    train_ds = SILVADataset(train_path, tokenizer, augment=True,  cache_size=0)
    val_ds   = SILVADataset(val_path,   tokenizer, augment=False, cache_size=0)

    logger.info(f"Dataset: Train={len(train_ds):,} | Val={len(val_ds):,}")

    # Weighted sampler hanya untuk train
    sampler = build_weighted_sampler(train_ds, rank_idx=4)  # bobot by Genus

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=sampler,                             # WeightedRandom, bukan shuffle
        num_workers=cfg.data.dataloader_num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True,                              # hindari batch kecil di akhir epoch
        persistent_workers=True,                    # worker tidak di-restart tiap epoch
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,         # val bisa batch lebih besar (no grad)
        shuffle=False,
        num_workers=cfg.data.dataloader_num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_dl, val_dl


def _build_dummy_loaders(cfg, tokenizer: KmerTokenizer):
    """Buat DataLoader dengan data dummy untuk testing pipeline tanpa data nyata."""
    from loguru import logger
    from torch.utils.data import TensorDataset

    logger.warning("Menggunakan DataLoader DUMMY untuk testing.")
    N = 200
    B = cfg.train.batch_size

    rng = torch.Generator().manual_seed(42)
    input_ids      = torch.randint(0, tokenizer.vocab_size, (N, cfg.data.max_seq_len))
    attention_mask = torch.ones(N, cfg.data.max_seq_len, dtype=torch.long)
    labels         = torch.randint(0, 10, (N, 6))

    ds = TensorDataset(input_ids, attention_mask, labels)

    def _dummy_collate(batch):
        inp, attn, lbl = zip(*batch)
        return {
            "seq_ids":        [f"dummy_{i}" for i in range(len(inp))],
            "input_ids":      torch.stack(inp),
            "attention_mask": torch.stack(attn),
            "labels":         torch.stack(lbl),
        }

    train_dl = DataLoader(ds, batch_size=B, shuffle=True,  collate_fn=_dummy_collate)
    val_dl   = DataLoader(ds, batch_size=B, shuffle=False, collate_fn=_dummy_collate)
    return train_dl, val_dl