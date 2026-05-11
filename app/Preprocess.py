"""
data/preprocess.py — SILVA v138 Curation Pipeline
==================================================
Modul ini menangani seluruh pipeline pra-pemrosesan data mentah SILVA:
  Stage 1: In-Silico PCR + quality filtering
  Stage 2: Taxonomic cleaning + LCA consensus + chimera removal
  Stage 3: Taxonomic Roll-up (long-tail mitigation)

Output akhir adalah file HDF5 yang siap dimuat secara lazy oleh dataset.py.

Penggunaan:
    python -m data.preprocess --config config.py
    # atau dari skrip lain:
    from data.preprocess import SILVACurator
    curator = SILVACurator(CFG)
    curator.run()
"""

import re
import pickle
import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    logger.warning("BioPython tidak terinstal. In-Silico PCR akan menggunakan regex sederhana.")


# ─────────────────────────────────────────────────────────────────────────────
# IUPAC AMBIGUITY TABLE
# ─────────────────────────────────────────────────────────────────────────────
IUPAC_AMBIGUOUS = set("NRYSWKMBDHV")  # basa ambigu (selain A/T/C/G)

IUPAC_REGEX = {
    "N": "[ATCG]", "R": "[AG]", "Y": "[CT]", "S": "[CG]",
    "W": "[AT]",   "K": "[GT]", "M": "[AC]", "B": "[CGT]",
    "D": "[AGT]",  "H": "[ACT]", "V": "[ACG]",
}


def primer_to_regex(primer: str, max_mismatch: int = 2) -> str:
    """
    Konversi primer DNA (dengan kode IUPAC) ke pola regex.
    Mismatch diakomodasi dengan membuat setiap posisi opsional
    menggunakan lookahead fuzzy sederhana.

    Parameter:
        primer       : sekuens primer 5'→3' (boleh mengandung kode IUPAC)
        max_mismatch : jumlah maksimum mismatch yang diizinkan

    Return:
        pola regex string untuk digunakan dengan re.search()
    """
    pattern = ""
    for base in primer.upper():
        if base in IUPAC_REGEX:
            pattern += IUPAC_REGEX[base]
        else:
            pattern += base
    # Catatan: implementasi mismatch penuh memerlukan regex-mismatch library.
    # Untuk produksi gunakan: `regex` package (pip install regex) atau VSEARCH --usearch_global.
    return pattern


def count_ambiguous_bases(seq: str) -> int:
    """Hitung jumlah basa ambigu IUPAC dalam sekuens."""
    return sum(1 for b in seq.upper() if b in IUPAC_AMBIGUOUS)


def apply_unk_token(seq: str, unk: str = "N") -> str:
    """Ganti basa ambigu dengan token [UNK] placeholder (tetap sebagai karakter N)."""
    return "".join(b if b not in IUPAC_AMBIGUOUS else unk for b in seq.upper())


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

TAX_RANKS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

def parse_silva_taxonomy(tax_string: str) -> Dict[str, str]:
    """
    Parsing string taksonomi SILVA (format: 'd__Bacteria;p__Firmicutes;...')
    menjadi dict rank → label.

    Parameter:
        tax_string : string taksonomi SILVA 7-level

    Return:
        dict {"Domain": "Bacteria", "Phylum": "Firmicutes", ...}
    """
    prefix_map = {"d": "Domain", "p": "Phylum", "c": "Class",
                  "o": "Order",  "f": "Family", "g": "Genus", "s": "Species"}
    result = {r: "" for r in TAX_RANKS}
    for token in tax_string.strip().split(";"):
        token = token.strip()
        if "__" in token:
            prefix, label = token.split("__", 1)
            rank = prefix_map.get(prefix.lower(), None)
            if rank:
                result[rank] = label.strip()
    return result


def is_uninformative(label: str, keywords: List[str]) -> bool:
    """Kembalikan True jika label mengandung kata kunci tidak informatif."""
    label_l = label.lower()
    return any(kw in label_l for kw in keywords)


def resolve_lca(tax_list: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Resolusi Lowest Common Ancestor (LCA) dari daftar anotasi taksonomi.
    Untuk setiap rank, semua label harus konsisten; jika tidak, rank tersebut
    dan semua rank di bawahnya dikosongkan.

    Parameter:
        tax_list : list dict taksonomi dari sekuens dalam satu cluster duplikat

    Return:
        dict taksonomi LCA
    """
    lca = {}
    for rank in TAX_RANKS:
        labels = set(t.get(rank, "") for t in tax_list if t.get(rank, ""))
        if len(labels) == 1:
            lca[rank] = labels.pop()
        else:
            # konflik → semua rank lebih halus dikosongkan
            break
    return lca


def taxonomic_rollup(
    tax: Dict[str, str],
    count: int,
    min_samples: int,
    rollup_max_rank: str = "Class",
) -> Dict[str, str]:
    """
    Terapkan Taxonomic Roll-up secara ITERATIF per rank.
    Jika jumlah sampel < min_samples, hapus rank terbawah dan periksa ulang;
    lanjutkan naik ke parent rank sampai count >= min_samples atau batas rank dicapai.

    Parameter:
        tax            : dict taksonomi satu sampel
        count          : jumlah sekuens untuk taxon terdalam saat ini
        min_samples    : threshold minimum sampel per takson
        rollup_max_rank: rank teratas yang boleh di-roll-up (tidak lewati ini)

    Return:
        dict taksonomi setelah roll-up
    """
    if count >= min_samples:
        return tax  # tidak perlu roll-up

    # Kumpulkan rank yang bisa di-roll-up (di bawah rollup_max_rank)
    rollable_ranks = []
    for rank in TAX_RANKS:  # Domain → Species
        if rank == rollup_max_rank:
            break
        rollable_ranks.append(rank)
    # rollable_ranks mencakup semua rank dari Domain s.d. sebelum rollup_max_rank
    # Rank yang bisa dihapus = rank di BAWAH rollup_max_rank (lebih halus)
    # Urutan hapus: dari paling halus (Species) naik ke atas
    fine_to_coarse = list(reversed(TAX_RANKS))  # Species → ... → Domain

    result = dict(tax)
    for rank in fine_to_coarse:
        if rank == rollup_max_rank:
            break  # jangan hapus rollup_max_rank atau lebih kasar
        result[rank] = ""  # hapus rank ini
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalLabelEncoder:
    """
    Encoder label hierarkis untuk semua 6 rank taksonomi.

    Metode:
        fit(tax_dicts)     : pelajari semua label unik dari data
        transform(tax)     : konversi dict taksonomi → array integer index
        inverse_transform  : konversi kembali ke label string
        save / load        : simpan / muat dari file PKL
    """

    def __init__(self, ranks: List[str] = None):
        self.ranks = ranks or TAX_RANKS[1:]  # Phylum → Species (skip Domain)
        self.label2idx: Dict[str, Dict[str, int]] = {r: {"<PAD>": 0, "<UNK>": 1} for r in self.ranks}
        self.idx2label: Dict[str, Dict[int, str]] = {r: {0: "<PAD>", 1: "<UNK>"} for r in self.ranks}

    def fit(self, tax_dicts: List[Dict[str, str]]) -> "HierarchicalLabelEncoder":
        """
        Pelajari semua label unik dari dataset.

        Parameter:
            tax_dicts : list dict taksonomi dari seluruh dataset

        Return:
            self (untuk chaining)
        """
        logger.info("Fitting HierarchicalLabelEncoder...")
        for rank in tqdm(self.ranks, desc="Encoding ranks"):
            labels = set()
            for tax in tax_dicts:
                lbl = tax.get(rank, "")
                if lbl:
                    labels.add(lbl)
            for lbl in sorted(labels):  # sorted untuk determinisme
                idx = len(self.label2idx[rank])
                self.label2idx[rank][lbl] = idx
                self.idx2label[rank][idx] = lbl
        logger.info(f"Vocab sizes: { {r: len(self.label2idx[r]) for r in self.ranks} }")
        return self

    def transform(self, tax: Dict[str, str]) -> np.ndarray:
        """
        Konversi dict taksonomi ke array index integer shape (6,).
        Label kosong → 0 (PAD), label tidak dikenal → 1 (UNK).
        """
        return np.array([
            self.label2idx[rank].get(tax.get(rank, ""), 0)
            for rank in self.ranks
        ], dtype=np.int32)

    def inverse_transform(self, indices: np.ndarray) -> Dict[str, str]:
        """Konversi array index kembali ke dict label string."""
        return {
            rank: self.idx2label[rank].get(int(idx), "<UNK>")
            for rank, idx in zip(self.ranks, indices)
        }

    def num_classes(self) -> Dict[str, int]:
        """Kembalikan jumlah kelas per rank."""
        return {r: len(self.label2idx[r]) for r in self.ranks}

    def decode(self, rank: str, idx: int) -> str:
        """
        Konversi index integer kembali ke label string untuk rank tertentu.

        Parameter:
            rank : nama rank ("Phylum", "Class", ..., "Species")
            idx  : integer index dari transform()

        Return:
            label string, atau "<PAD>" / "<UNK>" jika index khusus
        """
        return self.idx2label.get(rank, {}).get(int(idx), "<UNK>")

    def save(self, path: str):
        """Simpan encoder ke file PKL."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"LabelEncoder disimpan ke {path}")

    @staticmethod
    def load(path: str) -> "HierarchicalLabelEncoder":
        """Muat encoder dari file PKL."""
        with open(path, "rb") as f:
            enc = pickle.load(f)
        logger.info(f"LabelEncoder dimuat dari {path}")
        return enc


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE PIPELINE FUNCTIONS (dapat dipanggil langsung dari notebook)
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_lca(
    df: pd.DataFrame,
    ranks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Deduplikasi 100%-identik berbasis hash MD5 sequence, dengan resolusi LCA
    untuk cluster yang memiliki lebih dari satu anotasi taksonomi.

    Kolom 'sequence' wajib ada. Nilai NaN pada kolom rank diperlakukan sebagai
    string kosong saat resolusi LCA, lalu dikembalikan ke NaN.

    Parameter:
        df    : DataFrame dengan kolom 'seq_id', 'sequence', dan rank taksonomi.
        ranks : list rank yang ikut resolusi LCA (default TAX_RANKS).

    Return:
        DataFrame tanpa duplikat; satu wakil per sekuens unik dengan label LCA.
    """
    ranks = ranks or TAX_RANKS
    # Filter ke kolom yang benar-benar ada di DataFrame
    # (df bisa saja tidak punya kolom Domain jika dibuat dari ACTIVE_RANKS)
    ranks = [r for r in ranks if r in df.columns]
    n_before = len(df)

    # Hash setiap sekuens untuk pengelompokan cepat
    seq_hash = df["sequence"].apply(lambda s: hashlib.md5(s.encode()).hexdigest())
    df = df.copy()
    df["_seq_hash"] = seq_hash

    groups: List[dict] = []
    for _, grp in tqdm(
        df.groupby("_seq_hash", sort=False),
        desc="LCA dedup",
        total=df["_seq_hash"].nunique(),
    ):
        if len(grp) == 1:
            groups.append(grp.iloc[0].to_dict())
        else:
            # Gantikan NaN dengan "" agar resolve_lca bisa membandingkan
            tax_list = (
                grp[ranks]
                .fillna("")
                .to_dict("records")
            )
            lca = resolve_lca(tax_list)
            # Kembalikan string kosong ke NaN
            lca = {k: (v if v != "" else np.nan) for k, v in lca.items()}
            row = grp.iloc[0].to_dict()
            row.update(lca)
            groups.append(row)

    df_out = pd.DataFrame(groups).drop(columns=["_seq_hash"])
    n_removed = n_before - len(df_out)
    logger.info(
        f"Deduplikasi LCA: {n_removed:,} duplikat dihapus "
        f"({n_before:,} → {len(df_out):,} sekuens unik)"
    )
    return df_out.reset_index(drop=True)


def remove_chimeras(
    df: pd.DataFrame,
    threads: int = 4,
) -> pd.DataFrame:
    """
    Hapus chimeric sequences menggunakan VSEARCH --uchime3_denovo.
    Jika VSEARCH tidak ada di PATH, tampilkan peringatan dan kembalikan df asli
    tanpa error (graceful skip).

    Parameter:
        df      : DataFrame dengan kolom 'seq_id' dan 'sequence'.
        threads : jumlah thread VSEARCH (default 4).

    Return:
        DataFrame dengan chimera yang sudah dihapus, atau df asli jika VSEARCH
        tidak tersedia.
    """
    import shutil
    import subprocess
    import tempfile

    n_before = len(df)

    if shutil.which("vsearch") is None:
        logger.warning(
            "VSEARCH tidak ditemukan di PATH — chimera removal dilewati. "
            "Install dengan: conda install -c bioconda vsearch"
        )
        return df

    with tempfile.TemporaryDirectory() as tmpdir:
        in_fasta       = Path(tmpdir) / "input.fasta"
        out_nonchimera = Path(tmpdir) / "nonchimera.fasta"
        out_chimera    = Path(tmpdir) / "chimera.fasta"

        with open(in_fasta, "w") as fh:
            for _, row in df.iterrows():
                fh.write(f">{row['seq_id']}\n{row['sequence']}\n")

        cmd = [
            "vsearch",
            "--uchime3_denovo", str(in_fasta),
            "--nonchimeras",    str(out_nonchimera),
            "--chimeras",       str(out_chimera),
            "--threads",        str(threads),
            "--quiet",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                logger.warning(
                    f"VSEARCH gagal (returncode={result.returncode}): "
                    f"{result.stderr[:300]}. Chimera removal dilewati."
                )
                return df
        except subprocess.TimeoutExpired:
            logger.warning("VSEARCH timeout setelah 1 jam — chimera removal dilewati.")
            return df
        except Exception as exc:
            logger.warning(f"VSEARCH error: {exc} — chimera removal dilewati.")
            return df

        nonchimera_ids: set = set()
        with open(out_nonchimera) as fh:
            for line in fh:
                if line.startswith(">"):
                    nonchimera_ids.add(line[1:].strip().split()[0])

    df_out = df[df["seq_id"].isin(nonchimera_ids)].copy()
    n_removed = n_before - len(df_out)
    logger.info(
        f"Chimera removal: {n_removed:,} sekuens chimeric dihapus "
        f"({n_removed / max(n_before, 1) * 100:.1f}%). "
        f"Tersisa: {len(df_out):,}"
    )
    return df_out.reset_index(drop=True)


def rollup_taxa(
    df: pd.DataFrame,
    min_n: int,
    rollup_max_rank: str = "Class",
    ranks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Terapkan Taxonomic Roll-up secara iteratif pada DataFrame taksonomi.
    Setiap takson yang diwakili oleh < min_n sekuens dipromosikan ke rank parent
    (Species→Genus→…) sampai jumlah ≥ min_n atau batas rollup_max_rank tercapai.

    Kompatibel dengan NaN (pandas NA) maupun string kosong pada kolom rank.

    Parameter:
        df              : DataFrame dengan kolom-kolom nama rank taksonomi.
        min_n           : ambang minimum jumlah sekuens per takson.
        rollup_max_rank : rank terkasar yang boleh dijadikan target roll-up
                          (default "Class"). Rank ini sendiri TIDAK dihapus.
        ranks           : daftar rank yang diproses (default TAX_RANKS).

    Return:
        Salinan df dengan label yang telah di-roll-up.
    """
    ranks = ranks or TAX_RANKS
    df = df.copy()

    try:
        max_rank_idx = ranks.index(rollup_max_rank)
    except ValueError:
        max_rank_idx = ranks.index("Class")

    def _is_present(series: pd.Series) -> pd.Series:
        """True jika nilai bukan NaN dan bukan string kosong."""
        return series.notna() & (series.astype(str).str.strip() != "") & (series.astype(str) != "nan")

    for rank_idx in range(len(ranks) - 1, max_rank_idx, -1):
        rank        = ranks[rank_idx]
        parent_rank = ranks[rank_idx - 1]
        path_cols   = ranks[:rank_idx + 1]

        mask_complete = pd.concat(
            [_is_present(df[c]) for c in path_cols], axis=1
        ).all(axis=1)

        if not mask_complete.any():
            logger.info(f"  {rank}: tidak ada sekuens dengan path lengkap — lewati.")
            continue

        full_path = df.loc[mask_complete, path_cols].apply(
            lambda row: "|".join(row.values.astype(str)), axis=1
        )
        counts     = full_path.value_counts()
        rare_paths = set(counts[counts < min_n].index)

        if not rare_paths:
            logger.info(f"  Roll-up {rank:10s} → {parent_rank:10s}: tidak ada rare taxa")
            continue

        rare_mask = mask_complete & full_path.isin(rare_paths)
        n_rolled  = int(rare_mask.sum())
        for r in ranks[rank_idx:]:
            df.loc[rare_mask, r] = np.nan

        logger.info(
            f"  Roll-up {rank:10s} → {parent_rank:10s}: "
            f"{n_rolled:>7,} sekuens dipromosikan  "
            f"({len(rare_paths):,} rare taxa dihapus)"
        )

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CURATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SILVACurator:
    """
    Pipeline kurasi lengkap SILVA v138 → HDF5 siap latih.

    Tahap:
      1. Parsing FASTA + filter kualitas (In-Silico PCR, basa ambigu)
      2. Pembersihan label + chimera removal + LCA consensus
      3. Taxonomic Roll-up untuk long-tail mitigation
      4. Encoding label + stratified train/val split
      5. Penulisan ke HDF5 dengan chunking

    Penggunaan:
        curator = SILVACurator(CFG)
        curator.run()
        # file HDF5 tersimpan di CFG.paths.hdf5_train dan hdf5_val

    Parameter config relevan (di DataConfig):
        max_ambiguous_bases, min/max_amplicon_len,
        min_samples_per_taxon, uninformative_keywords,
        val_fraction, hdf5_chunk_size
    """

    def __init__(self, cfg):
        self.cfg = cfg
        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        self.primer_fwd_re = re.compile(
            primer_to_regex(cfg.data.primer_fwd, cfg.data.primer_max_mismatch),
            re.IGNORECASE,
        )
        self.primer_rev_re = re.compile(
            primer_to_regex(cfg.data.primer_rev, cfg.data.primer_max_mismatch),
            re.IGNORECASE,
        )

    # ── Stage 1: In-Silico PCR + Quality Filter ───────────────────────────
    def stage1_quality_filter(self) -> List[Tuple[str, str, str]]:
        """
        Baca FASTA SILVA, ekstrak amplikon V3-V4, filter basa ambigu.

        Return:
            list of (seq_id, amplicon_seq, raw_taxonomy_string)
        """
        logger.info("=== STAGE 1: In-Silico PCR + Quality Filtering ===")
        records = []
        n_total = n_too_short = n_too_long = n_ambig = n_no_primer = 0

        fasta_path = self.cfg.paths.silva_fasta
        if not Path(fasta_path).exists():
            logger.warning(f"FASTA tidak ditemukan: {fasta_path}. Menggunakan data demo.")
            return self._generate_demo_data(500)

        with open(fasta_path) as f:
            # Hitung total untuk tqdm
            total_lines = sum(1 for _ in f)

        logger.info(f"Membaca {fasta_path} ({total_lines:,} baris)...")

        with open(fasta_path) as f:
            pbar = tqdm(total=total_lines // 2, desc="Stage 1 — Filter", unit="seq")
            seq_id = tax_str = seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq:
                        # Proses rekam sebelumnya
                        result = self._process_record(seq_id, seq, tax_str, n_too_short, n_too_long, n_ambig, n_no_primer)
                        if result:
                            records.append(result)
                            n_total += 1
                        pbar.update(1)
                    parts = line[1:].split(" ", 1)
                    seq_id = parts[0]
                    tax_str = parts[1] if len(parts) > 1 else ""
                    seq = ""
                else:
                    seq += line
            # Proses rekam terakhir
            if seq:
                result = self._process_record(seq_id, seq, tax_str, n_total, n_too_short, n_too_long, n_ambig)
                if result:
                    records.append(result)
            pbar.close()

        logger.info(f"Stage 1 selesai: {len(records):,} amplikon lolos filter "
                    f"(dari ~{n_total:,} sekuens)")
        return records

    def _process_record(self, seq_id, seq, tax_str, *args):
        """Proses satu rekam: PCR, panjang, ambigu."""
        seq = seq.upper()

        # In-Silico PCR: cari primer fwd dan rev
        m_fwd = self.primer_fwd_re.search(seq)
        m_rev = self.primer_rev_re.search(seq)
        if not m_fwd or not m_rev:
            return None

        amplicon = seq[m_fwd.end():m_rev.start()]
        L = len(amplicon)

        # Filter panjang
        if L < self.cfg.data.min_amplicon_len or L > self.cfg.data.max_amplicon_len:
            return None

        # Filter basa ambigu
        if count_ambiguous_bases(amplicon) > self.cfg.data.max_ambiguous_bases:
            return None

        # Ganti sisa basa ambigu dengan N (token [UNK])
        amplicon = apply_unk_token(amplicon)
        return (seq_id, amplicon, tax_str)

    def _generate_demo_data(self, n: int = 500) -> List[Tuple[str, str, str]]:
        """Generate data demo sintetik jika FASTA tidak tersedia (untuk testing)."""
        logger.warning(f"Menggunakan {n} sekuens demo sintetik.")
        bases = list("ATCG")
        rng = np.random.default_rng(42)
        demo_taxa = [
            "d__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Lactobacillaceae;g__Lactobacillus;s__acidophilus",
            "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__coli",
            "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__fragilis",
        ]
        records = []
        for i in range(n):
            seq_len = rng.integers(380, 480)
            seq = "".join(rng.choice(bases, seq_len))
            tax = demo_taxa[i % len(demo_taxa)]
            records.append((f"demo_seq_{i}", seq, tax))
        return records

    # ── Stage 2: Taxonomic Cleaning + Dedup + LCA ─────────────────────────
    def stage2_taxonomy_clean(
        self, records: List[Tuple[str, str, str]]
    ) -> pd.DataFrame:
        """
        Bersihkan label tidak informatif, deduplikasi, resolusi LCA.
        Mendelegasikan ke :func:`deduplicate_lca` untuk logika dedup+LCA.

        Parameter:
            records : output dari stage1_quality_filter()

        Return:
            DataFrame dengan kolom: seq_id, sequence, Phylum, Class, ..., Species
        """
        logger.info("=== STAGE 2: Taxonomic Cleaning + Dedup + LCA ===")
        uninf_kw = self.cfg.data.uninformative_keywords
        rows = []

        for seq_id, seq, tax_str in tqdm(records, desc="Parse & clean labels"):
            tax = parse_silva_taxonomy(tax_str)
            for rank in TAX_RANKS:
                if is_uninformative(tax.get(rank, ""), uninf_kw):
                    for r in TAX_RANKS[TAX_RANKS.index(rank):]:
                        tax[r] = np.nan
                    break
            # string kosong → NaN
            for rank in TAX_RANKS:
                if tax.get(rank, "") == "":
                    tax[rank] = np.nan
            row = {"seq_id": seq_id, "sequence": seq}
            row.update(tax)
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Sebelum dedup: {len(df):,} sekuens")
        df_clean = deduplicate_lca(df, ranks=TAX_RANKS)
        logger.info(f"Setelah LCA dedup: {len(df_clean):,} sekuens unik")
        return df_clean

    # ── Stage 2b: Chimera Removal ─────────────────────────────────────────
    def stage2_chimera_removal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Hapus chimeric sequences menggunakan VSEARCH --uchime3_denovo.
        Mendelegasikan ke :func:`remove_chimeras`.

        Parameter:
            df : DataFrame dari stage2_taxonomy_clean()

        Return:
            DataFrame dengan chimera yang sudah dihapus (atau df asli jika VSEARCH tidak ada)
        """
        logger.info("=== STAGE 2b: Chimera Removal (VSEARCH) ===")
        return remove_chimeras(df, threads=4)

    # ── Stage 3: Taxonomic Roll-up ─────────────────────────────────────────
    def stage3_rollup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Terapkan Taxonomic Roll-up secara ITERATIF pada takson dengan < min_samples sekuens.
        Mendelegasikan ke :func:`rollup_taxa`.

        Parameter:
            df : DataFrame dari stage2_taxonomy_clean()

        Return:
            DataFrame dengan label yang telah di-roll-up (0 rare taxa tersisa)
        """
        logger.info("=== STAGE 3: Taxonomic Roll-up (iteratif) ===")
        return rollup_taxa(
            df,
            min_n=self.cfg.data.min_samples_per_taxon,
            rollup_max_rank=self.cfg.data.rollup_max_rank,
            ranks=TAX_RANKS,
        )

    # ── HDF5 Writer ───────────────────────────────────────────────────────
    def write_hdf5(
        self,
        df: pd.DataFrame,
        encoder: HierarchicalLabelEncoder,
        train_path: str,
        val_path: str,
    ):
        """
        Tulis DataFrame ke dua file HDF5 (train / val) dengan chunking.
        Struktur HDF5:
          /sequences  : array string sekuens DNA
          /labels     : array int32 shape (N, 6) — index per rank
          /seq_ids    : array string ID sekuens

        Parameter:
            df         : DataFrame lengkap hasil stage 3
            encoder    : HierarchicalLabelEncoder yang sudah di-fit
            train_path : path output train HDF5
            val_path   : path output val HDF5
        """
        logger.info("Menulis HDF5 (train + val)...")

        # Stratified split berdasarkan Genus (rank tengah)
        from sklearn.model_selection import train_test_split
        val_frac = self.cfg.data.val_fraction
        seed = self.cfg.data.random_seed

        # Buat stratify key dari Genus atau Phylum jika Genus kosong
        df["_strat_key"] = df["Genus"].where(df["Genus"] != "", df["Phylum"])
        df["_strat_key"] = df["_strat_key"].where(df["_strat_key"] != "", "Unknown")

        # Hapus kelas dengan < 2 sampel dari stratifikasi
        key_counts = df["_strat_key"].value_counts()
        valid_keys = key_counts[key_counts >= 2].index
        df_valid = df[df["_strat_key"].isin(valid_keys)]
        df_rare  = df[~df["_strat_key"].isin(valid_keys)]

        idx_train, idx_val = train_test_split(
            df_valid.index,
            test_size=val_frac,
            stratify=df_valid["_strat_key"],
            random_state=seed,
        )
        df_train = pd.concat([df.loc[idx_train], df_rare])  # rare → train semua
        df_val   = df.loc[idx_val]

        logger.info(f"Train: {len(df_train):,} | Val: {len(df_val):,}")

        for split_df, path in [
            (df_train, train_path),
            (df_val, val_path),
        ]:
            self._write_single_hdf5(split_df, encoder, path)

    def _write_single_hdf5(
        self, df: pd.DataFrame, encoder: HierarchicalLabelEncoder, path: str
    ):
        """Tulis satu split ke HDF5 dengan chunking untuk efisiensi I/O."""
        chunk = self.cfg.data.hdf5_chunk_size
        N = len(df)

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        seqs    = df["sequence"].values
        seq_ids = df["seq_id"].values if "seq_id" in df.columns else np.arange(N).astype(str)
        labels  = np.vstack([
            encoder.transform({r: row[r] for r in TAX_RANKS})
            for _, row in tqdm(df.iterrows(), total=N, desc=f"Encode labels → {Path(path).name}")
        ])

        with h5py.File(path, "w") as f:
            # Gunakan special_dtype untuk string variabel-panjang
            dt_str = h5py.special_dtype(vlen=str)

            f.create_dataset("sequences", data=seqs, dtype=dt_str, chunks=(chunk,))
            f.create_dataset("labels",    data=labels, dtype=np.int32, chunks=(chunk, 6))
            f.create_dataset("seq_ids",   data=seq_ids, dtype=dt_str, chunks=(chunk,))
            f.attrs["n_samples"] = N
            f.attrs["n_ranks"]   = 6
            f.attrs["ranks"]     = str(["Phylum","Class","Order","Family","Genus","Species"])

        logger.info(f"HDF5 ditulis: {path} ({N:,} sampel, {Path(path).stat().st_size/1e6:.1f} MB)")

    # ── Main Runner ───────────────────────────────────────────────────────
    def run(self):
        """
        Jalankan pipeline kurasi lengkap dari awal hingga akhir.
        Simpan semua artefak intermediate ke disk.
        """
        # Stage 1
        records = self.stage1_quality_filter()

        # Stage 2
        df_clean = self.stage2_taxonomy_clean(records)

        # Stage 2b: Chimera removal (graceful skip jika VSEARCH tidak ada)
        df_clean = self.stage2_chimera_removal(df_clean)

        # Fit label encoder sebelum roll-up (agar vocabulary mencakup semua label asli)
        tax_dicts = df_clean[TAX_RANKS].to_dict("records")
        encoder = HierarchicalLabelEncoder()
        encoder.fit(tax_dicts)
        encoder.save(self.cfg.paths.label_encoder)

        # Stage 3
        df_rolled = self.stage3_rollup(df_clean)

        # Tulis HDF5
        self.write_hdf5(
            df_rolled, encoder,
            self.cfg.paths.hdf5_train,
            self.cfg.paths.hdf5_val,
        )

        # Simpan statistik preprocessing ke CSV
        stats = {
            "stage": ["raw", "after_stage1", "after_stage2", "after_stage3"],
            "n_sequences": [
                "~2,224,740",
                len(records),
                len(df_clean),
                len(df_rolled),
            ],
        }
        stats_path = Path(self.cfg.paths.metric_dir) / "preprocessing_stats.csv"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stats).to_csv(stats_path, index=False)
        logger.info(f"Statistik preprocessing disimpan ke {stats_path}")
        logger.success("=== Pipeline kurasi selesai ===")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────�