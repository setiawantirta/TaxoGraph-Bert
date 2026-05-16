"""
data_acquisition.py
-------------------
Modul untuk akuisisi dataset eksternal yang dibutuhkan untuk:
  4.1 Mockrobiota benchmark (komunitas mikrobial sintetik/tiruan)
  4.2 NCBI temporal hold-out OOD (sekuens 16S yang dipublikasikan setelah 2021)

Fungsi utama:
  - download_mockrobiota(output_dir, mock_ids)
  - fetch_ncbi_temporal_holdout(output_fasta, label_enc, n, cutoff)
  - load_mockrobiota_dataset(mock_dir, mock_id)
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MOCKROBIOTA
# ─────────────────────────────────────────────────────────────────────────────

# Dataset 16S yang tersedia di mockrobiota (AWS S3)
MOCKROBIOTA_16S_IDS = [1, 2, 3, 4, 5, 7, 8, 16, 20, 21, 22, 23]

# URL base untuk mockrobiota di S3
MOCKROBIOTA_S3_BASE = (
    "https://s3-us-west-2.amazonaws.com/mockrobiota/latest/"
    "mock-{mock_id}/mock-forward-read.fastq.gz"
)

# URL file expected-sequence.tsv per mock (ground truth taksonomi)
MOCKROBIOTA_EXPECTED_BASE = (
    "https://s3-us-west-2.amazonaws.com/mockrobiota/latest/"
    "mock-{mock_id}/expected-taxonomy.tsv"
)


def download_mockrobiota(
    output_dir: str | Path,
    mock_ids: Optional[List[int]] = None,
    retry: int = 3,
    delay: float = 2.0,
    streaming: bool = True,
    local_fastq_paths: Optional[dict] = None,
    max_reads: int = 2000,
) -> dict[int, Path]:
    """
    Unduh dataset mockrobiota dari AWS S3.

    Mengunduh `mock-forward-read.fastq.gz` dan `expected-taxonomy.tsv`
    untuk setiap dataset yang diminta.

    Parameter:
        output_dir         : direktori lokal tempat file disimpan
                             Struktur: output_dir/mock-{N}/mock-forward-read.fastq
                                       output_dir/mock-{N}/expected-taxonomy.tsv
        mock_ids           : list ID dataset (default: semua 16S = MOCKROBIOTA_16S_IDS)
        retry              : jumlah percobaan ulang jika download gagal
        delay              : jeda antar percobaan ulang (detik)
        streaming          : jika True (default), streaming langsung dari URL tanpa
                             menyimpan file .gz penuh ke disk — hemat ruang penyimpanan
                             (file .fastq kecil ~max_reads baris yang disimpan).
                             jika False, unduh dan decompress file penuh (~10 GB).
        local_fastq_paths  : dict {mock_id: path} — path ke file .fastq atau .fastq.gz
                             yang sudah diunduh manual. Jika disediakan untuk mock_id
                             tertentu, download dari URL di-skip untuk ID tersebut.
                             Contoh: {1: "/kaggle/input/mockrobiota/mock-1.fastq"}
        max_reads          : jumlah reads yang disimpan saat streaming (default 2000).
                             Tidak berpengaruh jika streaming=False atau file sudah ada.

    Return:
        dict {mock_id: Path} — path ke folder setiap mock dataset
    """
    if mock_ids is None:
        mock_ids = MOCKROBIOTA_16S_IDS
    if local_fastq_paths is None:
        local_fastq_paths = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[int, Path] = {}

    for mid in mock_ids:
        mock_folder = output_dir / f"mock-{mid}"
        mock_folder.mkdir(parents=True, exist_ok=True)

        # ── Tentukan sumber FASTQ ────────────────────────────────────────────
        fastq_out = mock_folder / "mock-forward-read.fastq"

        if fastq_out.exists():
            logger.info(f"[mockrobiota] mock-{mid} FASTQ sudah ada, skip download.")

        elif mid in local_fastq_paths:
            # Fallback: gunakan file yang sudah diunduh manual
            local_path = Path(local_fastq_paths[mid])
            if not local_path.exists():
                raise FileNotFoundError(
                    f"[mockrobiota] local_fastq_paths[{mid}] tidak ditemukan: {local_path}"
                )
            if local_path.suffix == ".gz":
                # Decompress streaming dari file lokal .gz
                logger.info(f"[mockrobiota] mock-{mid} decompress dari file lokal: {local_path}")
                sequences = _read_fastq_sequences_from_gz_file(
                    local_path, max_reads=max_reads
                )
            else:
                # Salin file lokal .fastq
                import shutil
                logger.info(f"[mockrobiota] mock-{mid} salin dari file lokal: {local_path}")
                shutil.copy2(local_path, fastq_out)
                sequences = None  # file sudah di-copy, baca saat diperlukan
            if sequences is not None:
                _write_fastq_sequences(fastq_out, sequences)
                logger.info(f"[mockrobiota] mock-{mid} {len(sequences)} reads ditulis dari file lokal.")

        elif streaming:
            # Mode streaming: baca langsung dari URL, simpan hanya max_reads reads
            fastq_gz_url = MOCKROBIOTA_S3_BASE.format(mock_id=mid)
            logger.info(
                f"[mockrobiota] mock-{mid} streaming {max_reads} reads dari URL "
                f"(tanpa simpan file .gz penuh)..."
            )
            sequences = _stream_sequences_from_gz_url(
                fastq_gz_url, max_reads=max_reads, retry=retry, delay=delay
            )
            _write_fastq_sequences(fastq_out, sequences)
            logger.info(f"[mockrobiota] mock-{mid} {len(sequences)} reads disimpan ke {fastq_out}")

        else:
            # Mode full download: unduh dan decompress file penuh (~10 GB)
            fastq_gz_url = MOCKROBIOTA_S3_BASE.format(mock_id=mid)
            _download_and_decompress(fastq_gz_url, fastq_out, retry=retry, delay=delay)

        # ── Unduh expected-taxonomy.tsv ─────────────────────────────────────
        tax_url  = MOCKROBIOTA_EXPECTED_BASE.format(mock_id=mid)
        tax_out  = mock_folder / "expected-taxonomy.tsv"

        if not tax_out.exists():
            _download_file(tax_url, tax_out, retry=retry, delay=delay)
        else:
            logger.info(f"[mockrobiota] mock-{mid} taxonomy sudah ada, skip download.")

        downloaded[mid] = mock_folder
        logger.info(f"[mockrobiota] mock-{mid} siap: {mock_folder}")

    return downloaded


def load_mockrobiota_dataset(
    mock_dir: str | Path,
    mock_id: int,
    max_reads: int | None = 2000,
) -> dict:
    """
    Muat satu dataset mockrobiota dari folder lokal.

    Parameter:
        mock_dir  : direktori root mockrobiota (sama dengan `output_dir` di download_mockrobiota)
        mock_id   : ID dataset mockrobiota (misal: 1, 2, 3, ...)
        max_reads : batas jumlah reads yang dimuat ke RAM (default 2000).
                    None = tidak ada batas (hati-hati: bisa OOM untuk dataset besar).

    Return:
        dict dengan key:
          "sequences"  : list[str] — sekuens DNA dari FASTQ (reads)
          "taxonomy"   : list[dict] — expected taxonomy per OTU (dari TSV)
          "mock_id"    : int
    """
    import csv

    mock_folder = Path(mock_dir) / f"mock-{mock_id}"
    fastq_path  = mock_folder / "mock-forward-read.fastq"
    fastq_gz_path = mock_folder / "mock-forward-read.fastq.gz"
    tax_path    = mock_folder / "expected-taxonomy.tsv"

    if not fastq_path.exists() and not fastq_gz_path.exists():
        raise FileNotFoundError(
            f"FASTQ tidak ditemukan: {fastq_path} (atau {fastq_gz_path}). "
            "Jalankan download_mockrobiota() terlebih dahulu, atau "
            "letakkan file .fastq/.fastq.gz secara manual di folder tersebut."
        )

    # Baca sekuens dari FASTQ secara streaming (1 baris per iterasi → O(1) memori)
    # FASTQ format: setiap record = 4 baris; baris ke-2 (index 1) = sequence
    # Fallback: baca langsung dari .fastq.gz jika .fastq tidak ada
    sequences = []
    if fastq_path.exists():
        with open(fastq_path, "r", errors="replace") as fh:
            line_num = 0
            for line in fh:
                if line_num % 4 == 1:  # baris sequence dalam record FASTQ
                    seq = line.strip()
                    if seq:
                        sequences.append(seq)
                        if max_reads is not None and len(sequences) >= max_reads:
                            break
                line_num += 1
    else:
        # Baca langsung dari .fastq.gz tanpa decompress penuh ke disk
        sequences = _read_fastq_sequences_from_gz_file(fastq_gz_path, max_reads=max_reads)

    # Baca expected taxonomy
    taxonomy = []
    if tax_path.exists():
        with open(tax_path, "r", newline="", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                taxonomy.append(dict(row))

    return {"sequences": sequences, "taxonomy": taxonomy, "mock_id": mock_id}


# ─────────────────────────────────────────────────────────────────────────────
# NCBI TEMPORAL HOLD-OUT
# ─────────────────────────────────────────────────────────────────────────────

# Primer 16S V3-V4 untuk in-silico PCR
PRIMER_FWD = "CCTACGGGNGGCWGCAG"   # 341F
PRIMER_REV = "GACTACHVGGGTATCTAATCC"  # 805R (reverse)

# BioProjects SILVA v138 source projects
SILVA_BIOPROJECTS = ["PRJNA33175", "PRJNA33317"]


def fetch_ncbi_temporal_holdout(
    output_fasta: str | Path,
    label_enc,
    n: int = 50,
    cutoff: str = "2021-01-01",
    email: str = "user@example.com",
) -> Path:
    """
    Ambil sekuens 16S dari NCBI yang dipublikasikan SETELAH tanggal cutoff
    dan genus-nya TIDAK ada di SILVA v138 (OOD ground truth).

    Gunakan Biopython Entrez untuk query BioProjects SILVA (PRJNA33175, PRJNA33317).
    Filter genus yang tidak ada di label_enc. In-silico PCR V3-V4 (primer 341F/805R).

    Parameter:
        output_fasta : path file FASTA output
        label_enc    : HierarchicalLabelEncoder yang sudah di-fit (untuk cek genus OOD)
        n            : jumlah sekuens OOD yang diinginkan
        cutoff       : tanggal publikasi minimum (YYYY-MM-DD)
        email        : email untuk Entrez (diwajibkan oleh NCBI)

    Return:
        Path ke file FASTA yang ditulis
    """
    try:
        from Bio import Entrez, SeqIO
    except ImportError:
        raise ImportError(
            "Biopython diperlukan: pip install biopython"
        )

    Entrez.email = email

    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    # Ambil genus yang sudah diketahui di SILVA (in-distribution)
    genus_rank   = "Genus"
    known_genera: set[str] = set()
    if hasattr(label_enc, "label2idx") and genus_rank in label_enc.label2idx:
        known_genera = set(label_enc.label2idx[genus_rank].keys()) - {"<PAD>", "<UNK>"}

    logger.info(f"[ncbi] Genus in-distribution: {len(known_genera)} entries")

    # Query Entrez untuk 16S rRNA yang dipublikasi setelah cutoff
    # Format tanggal Entrez: YYYY/MM/DD
    cutoff_entrez = cutoff.replace("-", "/")
    query = (
        f'("16S ribosomal RNA"[Title] OR "16S rRNA"[Title]) '
        f'AND ("{cutoff_entrez}"[Publication Date] : "3000"[Publication Date]) '
        f'AND biomol_rrna[PROP]'
    )

    logger.info(f"[ncbi] Query Entrez: {query[:100]}...")

    handle    = Entrez.esearch(db="nucleotide", term=query, retmax=n * 10, usehistory="y")
    search_res = Entrez.read(handle)
    handle.close()

    webenv    = search_res["WebEnv"]
    query_key = search_res["QueryKey"]
    total     = int(search_res["Count"])
    logger.info(f"[ncbi] Total results: {total}")

    if total == 0:
        logger.warning("[ncbi] Tidak ada hasil dari Entrez. Pastikan query benar.")
        output_fasta.write_text("")
        return output_fasta

    # Fetch sekuens dalam batch
    written  = 0
    batch_sz = 50
    records_ood = []

    for start in range(0, min(total, n * 10), batch_sz):
        if written >= n:
            break
        try:
            handle = Entrez.efetch(
                db="nucleotide",
                rettype="fasta",
                retmode="text",
                retstart=start,
                retmax=batch_sz,
                webenv=webenv,
                query_key=query_key,
            )
            batch_text = handle.read()
            handle.close()
        except Exception as exc:
            logger.warning(f"[ncbi] Batch {start} gagal: {exc}")
            time.sleep(2.0)
            continue

        for record in SeqIO.parse(io.StringIO(batch_text), "fasta"):
            seq_str = str(record.seq).upper()

            # In-silico PCR: ekstrak amplikon V3-V4
            amplicon = _insilico_pcr(seq_str, PRIMER_FWD, PRIMER_REV)
            if amplicon is None:
                continue

            # Cek apakah genus ada di deskripsi record (heuristik: cari kata sebelum sp./cf.)
            genus = _extract_genus_from_description(record.description)
            if genus and genus in known_genera:
                continue  # in-distribution, skip

            record.seq     = type(record.seq)(amplicon)
            records_ood.append(record)
            written += 1

            if written >= n:
                break

        time.sleep(0.35)  # NCBI rate limit: max 3 req/s

    with open(output_fasta, "w") as fh:
        SeqIO.write(records_ood, fh, "fasta")

    logger.info(f"[ncbi] {len(records_ood)} sekuens OOD ditulis ke {output_fasta}")
    return output_fasta


# ─────────────────────────────────────────────────────────────────────────────
# HELPER PRIVAT
# ─────────────────────────────────────────────────────────────────────────────

def _stream_sequences_from_gz_url(
    url: str,
    max_reads: int = 2000,
    retry: int = 3,
    delay: float = 2.0,
) -> list:
    """
    Baca sekuens FASTQ langsung dari URL .gz tanpa menyimpan file ke disk.

    Menggunakan streaming HTTP + gzip sehingga hanya data yang dibutuhkan
    (max_reads pertama) yang diunduh — koneksi ditutup segera setelah terpenuhi.
    Hemat ruang disk secara drastis (menghindari download file ~10 GB).

    Return:
        list[str] — list sekuens DNA
    """
    for attempt in range(1, retry + 1):
        try:
            logger.info(f"Streaming FASTQ dari URL ({attempt}/{retry}): {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "taxograph-bert/1.0 data-acquisition"},
            )
            sequences = []
            with urllib.request.urlopen(req, timeout=300) as resp:
                with gzip.open(resp, "rt", errors="replace") as gz_fh:
                    line_num = 0
                    for line in gz_fh:
                        if line_num % 4 == 1:  # baris ke-2 setiap record FASTQ = sequence
                            seq = line.strip()
                            if seq:
                                sequences.append(seq)
                                if len(sequences) >= max_reads:
                                    break
                        line_num += 1
            logger.info(f"Streaming selesai: {len(sequences)} reads diperoleh dari {url}")
            return sequences
        except Exception as exc:
            logger.warning(f"Gagal streaming {url}: {exc}")
            if attempt < retry:
                time.sleep(delay)
    raise RuntimeError(f"Gagal streaming setelah {retry} percobaan: {url}")


def _read_fastq_sequences_from_gz_file(
    gz_path: Path,
    max_reads: Optional[int] = 2000,
) -> list:
    """
    Baca sekuens FASTQ dari file .fastq.gz lokal tanpa decompress penuh ke disk.

    Return:
        list[str] — list sekuens DNA
    """
    sequences = []
    with gzip.open(gz_path, "rt", errors="replace") as gz_fh:
        line_num = 0
        for line in gz_fh:
            if line_num % 4 == 1:
                seq = line.strip()
                if seq:
                    sequences.append(seq)
                    if max_reads is not None and len(sequences) >= max_reads:
                        break
            line_num += 1
    return sequences


def _write_fastq_sequences(dest: Path, sequences: list) -> None:
    """
    Tulis list sekuens ke file .fastq minimal (hanya sequence lines, format FASTQ sederhana).
    Digunakan untuk menyimpan hasil streaming agar tidak perlu re-download.
    """
    with open(dest, "w") as fh:
        for i, seq in enumerate(sequences):
            fh.write(f"@read_{i}\n{seq}\n+\n{'I' * len(seq)}\n")


def _download_file(
    url: str,
    dest: Path,
    retry: int = 3,
    delay: float = 2.0,
) -> None:
    """Unduh file dari URL ke dest; retry jika gagal."""
    for attempt in range(1, retry + 1):
        try:
            logger.info(f"Mengunduh ({attempt}/{retry}): {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "taxograph-bert/1.0 data-acquisition"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return
        except Exception as exc:
            logger.warning(f"Gagal download {url}: {exc}")
            if attempt < retry:
                time.sleep(delay)
    raise RuntimeError(f"Gagal mengunduh setelah {retry} percobaan: {url}")


def _download_and_decompress(
    url: str,
    dest: Path,
    retry: int = 3,
    delay: float = 2.0,
    chunk_size: int = 65536,
) -> None:
    """Unduh file .gz dan decompress ke dest secara streaming (chunk-by-chunk)."""
    for attempt in range(1, retry + 1):
        try:
            logger.info(f"Mengunduh+decompress ({attempt}/{retry}): {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "taxograph-bert/1.0 data-acquisition"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                with gzip.open(resp, "rb") as gz_stream:
                    with open(dest, "wb") as out_fh:
                        while True:
                            chunk = gz_stream.read(chunk_size)
                            if not chunk:
                                break
                            out_fh.write(chunk)
            return
        except Exception as exc:
            logger.warning(f"Gagal download+decompress {url}: {exc}")
            # Hapus file partial agar tidak terpakai pada percobaan berikutnya
            if dest.exists():
                dest.unlink()
            if attempt < retry:
                time.sleep(delay)
    raise RuntimeError(f"Gagal mengunduh setelah {retry} percobaan: {url}")


def _insilico_pcr(
    seq: str,
    primer_fwd: str,
    primer_rev: str,
    max_mismatch: int = 2,
) -> Optional[str]:
    """
    Lakukan in-silico PCR pada sekuens DNA.
    Cari primer forward dan reverse, ekstrak amplikon di antaranya.
    Primer matching menggunakan exact substring (karena IUPAC primer sudah dihandle oleh desain).

    Parameter:
        seq          : sekuens DNA (uppercase)
        primer_fwd   : sekuens primer forward
        primer_rev   : sekuens primer reverse (akan dicari sebagai reverse complement)
        max_mismatch : jumlah mismatch maksimum (belum diimplementasikan, pakai exact)

    Return:
        str amplikon atau None jika primer tidak ditemukan
    """
    def rev_comp(s: str) -> str:
        comp = {"A": "T", "T": "A", "G": "C", "C": "G",
                "N": "N", "W": "W", "S": "S", "R": "Y",
                "Y": "R", "K": "M", "M": "K", "B": "V",
                "D": "H", "H": "D", "V": "B"}
        return "".join(comp.get(b, "N") for b in reversed(s))

    rev_primer = rev_comp(primer_rev)

    fwd_pos = seq.find(primer_fwd[:8])  # match 8 bp pertama dari forward primer
    if fwd_pos < 0:
        return None

    rev_pos = seq.find(rev_primer[:8], fwd_pos + len(primer_fwd))
    if rev_pos < 0:
        return None

    amplicon = seq[fwd_pos + len(primer_fwd): rev_pos]
    # Amplikon V3-V4 ~ 400-450 bp
    if len(amplicon) < 200 or len(amplicon) > 600:
        return None

    return amplicon


def _extract_genus_from_description(description: str) -> Optional[str]:
    """
    Ekstrak nama genus dari deskripsi FASTA header (heuristik).
    Format NCBI: ">accession.version Genus species ..."
    Ambil kata kedua dari deskripsi (setelah accession).
    """
    parts = description.split()
    if len(parts) >= 3:
        # parts[0] = accession, parts[1] = genus, parts[2] = species
        genus = parts[1]
        if genus[0].isupper() and genus[1:].islower():
            return genus
    return None
