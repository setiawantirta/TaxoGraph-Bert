"""
models/hyperbolic.py — Operasi Geometri Hiperbolik & HyTaxGNN
=============================================================
Implementasi Poincaré ball dan Hyperbolic Graph Convolutional Network (HGCN)
untuk embedding pohon taksonomi SILVA ke dalam ruang hiperbolik.

Komponen:
  PoincareBall   : operasi geometri pada Poincaré ball (jarak, Möbius add, exp/log map)
  HGCNLayer      : satu layer Hyperbolic GCN (message passing di ruang hiperbolik)
  HyTaxGNN       : jaringan lengkap (3 layer HGCN) untuk embedding taksonomi

Referensi:
  Chami et al. (2019) "Hyperbolic Graph Convolutional Neural Networks" NeurIPS
  Nickel & Kiela (2017) "Poincaré Embeddings for Learning Hierarchical Representations"
  geoopt library: https://github.com/geoopt/geoopt

Penggunaan:
    from models.hyperbolic import HyTaxGNN, PoincareBall
    gnn = HyTaxGNN(in_dim=32, hidden_dim=128, n_layers=3, curvature=1.0)
    node_emb = gnn(node_features, edge_index)  # shape (N_nodes, 128) di Poincaré space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# POINCARÉ BALL OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class PoincareBall:
    """
    Kumpulan operasi geometri pada Poincaré ball B^n_c dengan kelengkungan c.

    Poincaré ball: { x ∈ ℝ^n : c·||x||² < 1 }
    Jarak geodesik tumbuh eksponensial dari pusat → ideal untuk tree embedding.

    Semua operasi menjaga tensor tetap di dalam bola (clamp ke batas).

    Parameter:
        c : kelengkungan (curvature), nilai positif (default 1.0)
            c lebih besar → ruang "lebih melengkung" → representasi hierarki lebih tajam
    """

    def __init__(self, c: float = 1.0):
        self.c = c
        self.eps = 1e-5  # epsilon numerik untuk stabilitas

    @property
    def radius(self) -> float:
        """Radius bola = 1/sqrt(c)"""
        return 1.0 / (self.c ** 0.5)

    def proj(self, x: Tensor) -> Tensor:
        """
        Proyeksikan tensor ke dalam Poincaré ball (clamp jika di luar batas).

        Parameter:
            x : tensor arbiter shape (..., d)

        Return:
            x yang dijamin berada di dalam bola
        """
        max_norm = (1.0 - self.eps) / (self.c ** 0.5)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * scale

    def mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Möbius addition: x ⊕_c y (operasi "penjumlahan" di ruang hiperbolik).
        Pengganti penjumlahan Euclidean untuk feature fusion.

        x ⊕_c y = [(1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y]
                  / [1 + 2c<x,y> + c²||x||²||y||²]

        Parameter:
            x, y : tensor shape (..., d), keduanya harus ada di Poincaré ball

        Return:
            tensor hasil Möbius addition, shape (..., d)
        """
        c = self.c
        xy = (x * y).sum(dim=-1, keepdim=True)  # dot product <x, y>
        x2 = (x * x).sum(dim=-1, keepdim=True)  # ||x||²
        y2 = (y * y).sum(dim=-1, keepdim=True)  # ||y||²

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2

        return self.proj(num / denom.clamp(min=self.eps))

    def expmap0(self, v: Tensor) -> Tensor:
        """
        Exponential map dari origin (0) ke Poincaré ball.
        Digunakan untuk memetakan vektor tangent space → Poincaré ball.
        Ini adalah operasi kunci untuk memproyeksikan output linear ke dalam ruang hiperbolik.

        expmap_0(v) = tanh(√c · ||v||) · v / (√c · ||v||)

        Parameter:
            v : vektor di tangent space (Euclidean), shape (..., d)

        Return:
            titik di Poincaré ball, shape (..., d)
        """
        sqrt_c = self.c ** 0.5
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.proj(torch.tanh(sqrt_c * norm_v) * v / (sqrt_c * norm_v))

    def logmap0(self, y: Tensor) -> Tensor:
        """
        Logarithmic map dari origin (0) — kebalikan dari expmap0.
        Digunakan untuk memetakan titik Poincaré ball → tangent space (Euclidean).
        Diperlukan untuk operasi linear (matriks perkalian) di hyperbolic GNN.

        logmap_0(y) = atanh(√c · ||y||) · y / (√c · ||y||)

        Parameter:
            y : titik di Poincaré ball, shape (..., d)

        Return:
            vektor di tangent space, shape (..., d)
        """
        sqrt_c = self.c ** 0.5
        norm_y = y.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        # Pastikan input atanh valid: |x| < 1
        norm_y_safe = (sqrt_c * norm_y).clamp(max=1 - self.eps)
        return torch.atanh(norm_y_safe) * y / (sqrt_c * norm_y)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Geodesic distance antara dua titik di Poincaré ball.
        d_c(x, y) = (2/√c) · atanh(√c · ||(-x) ⊕_c y||)

        Parameter:
            x, y : tensor shape (..., d)

        Return:
            jarak scalar shape (...)
        """
        sqrt_c = self.c ** 0.5
        add_xy = self.mobius_add(-x, y)
        norm = add_xy.norm(dim=-1).clamp(max=1 - self.eps)
        return (2.0 / sqrt_c) * torch.atanh(sqrt_c * norm)

    def pairwise_dist(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Hitung semua jarak pairwise antara baris X (N,d) dan baris Y (M,d).
        Digunakan untuk OOD scoring (min dist ke semua leaf nodes).

        Return:
            tensor shape (N, M) — dist[i, j] = d(X[i], Y[j])
        """
        N = X.shape[0]
        M = Y.shape[0]
        # Expand untuk vectorized computation
        X_exp = X.unsqueeze(1).expand(N, M, -1)  # (N, M, d)
        Y_exp = Y.unsqueeze(0).expand(N, M, -1)  # (N, M, d)
        return self.dist(X_exp, Y_exp)            # (N, M)


# ─────────────────────────────────────────────────────────────────────────────
# HYPERBOLIC GCN LAYER
# ─────────────────────────────────────────────────────────────────────────────

class HGCNLayer(nn.Module):
    """
    Satu layer Hyperbolic Graph Convolutional Network (HGCN).

    Alur message passing hiperbolik:
      1. logmap0: proyeksikan embedding hiperbolik → tangent space (Euclidean)
      2. Linear transform: W·h + b di tangent space
      3. Aggregasi tetangga (mean pooling) di tangent space
      4. expmap0: proyeksikan kembali → Poincaré ball
      5. Aktivasi hiperbolik (non-linearity via tangent space)

    Parameter:
        in_dim   : dimensi input
        out_dim  : dimensi output
        ball     : instance PoincareBall
        dropout  : dropout rate di tangent space

    Input:
        x          : node embeddings di Poincaré ball, shape (N, in_dim)
        edge_index : tensor koneksi edge shape (2, E) — COO format
                     edge_index[0] = source nodes, edge_index[1] = target nodes

    Output:
        tensor node embeddings baru di Poincaré ball, shape (N, out_dim)
    """

    def __init__(
        self, in_dim: int, out_dim: int, ball: PoincareBall, dropout: float = 0.1
    ):
        super().__init__()
        self.ball    = ball
        self.linear  = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Inisialisasi dengan skala kecil agar output awal dekat dengan origin
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass satu layer HGCN.

        Parameter:
            x          : (N, in_dim) — node features di Poincaré ball
            edge_index : (2, E) — edge connectivity

        Return:
            (N, out_dim) — updated node embeddings di Poincaré ball
        """
        N = x.shape[0]

        # Step 1: Poincaré → tangent space (Euclidean)
        h = self.ball.logmap0(x)                  # (N, in_dim)

        # Step 2: Linear transform di tangent space
        h = self.dropout(h)
        h_trans = self.linear(h)                  # (N, out_dim)

        # Step 3: Message passing — backend-agnostic (MPS / CUDA / CPU)
        # Menggunakan scatter_add_ agar kompatibel dengan semua backend PyTorch.
        # torch.sparse_coo_tensor + torch.sparse.mm TIDAK didukung pada MPS (Apple Silicon).
        src, dst = edge_index[0], edge_index[1]
        out_dim = h_trans.shape[1]

        # Kumpulkan pesan: untuk tiap edge (src→dst), akumulasi fitur src ke bucket dst
        agg = torch.zeros(N, out_dim, device=x.device, dtype=h_trans.dtype)
        idx = dst.unsqueeze(1).expand(-1, out_dim)    # (E, out_dim)
        agg.scatter_add_(0, idx, h_trans[src])         # sum messages per dst node

        # Hitung in-degree tiap node untuk normalisasi mean
        deg = torch.zeros(N, device=x.device, dtype=h_trans.dtype)
        ones_e = torch.ones(dst.shape[0], device=x.device, dtype=h_trans.dtype)
        deg.scatter_add_(0, dst, ones_e)
        deg = deg.clamp(min=1.0)                      # hindari division by zero

        agg = agg / deg.unsqueeze(1)                  # (N, out_dim) — mean aggregation

        # Residual: tambahkan fitur node sendiri
        h_agg = h_trans + agg                     # (N, out_dim)

        # Step 4: Tangent space → Poincaré ball
        x_out = self.ball.expmap0(h_agg)          # (N, out_dim)

        return x_out


# ─────────────────────────────────────────────────────────────────────────────
# FULL HYPERBOLIC TAXONOMY GNN (HyTaxGNN)
# ─────────────────────────────────────────────────────────────────────────────

class HyTaxGNN(nn.Module):
    """
    Hyperbolic Taxonomy Graph Neural Network.
    Embed seluruh pohon taksonomi SILVA ke dalam Poincaré ball.

    Arsitektur:
      - Embedding layer: one-hot node features → ℝ^in_dim → Poincaré ball
      - 3 × HGCNLayer dengan skip-connection di tangent space
      - Output: node embeddings di Poincaré ball, shape (N_nodes, out_dim)

    Node akar (Domain) berada dekat origin; leaf species nodes dekat boundary.
    Ini mencerminkan sifat alami ruang hiperbolik: volume tumbuh eksponensial.

    Parameter:
        num_nodes  : jumlah node dalam taxonomy graph (total semua takson)
        in_dim     : dimensi embedding awal per node
        hidden_dim : dimensi embedding hiperbolik (output setiap layer HGCN)
        n_layers   : jumlah layer HGCN (default 3)
        curvature  : kelengkungan Poincaré ball c (default 1.0)
        dropout    : dropout rate

    Penggunaan:
        gnn = HyTaxGNN(num_nodes=50000, in_dim=64, hidden_dim=128, n_layers=3)
        # edge_index: tensor (2, E) dari taxonomy DAG
        node_emb = gnn(edge_index)  # shape (50000, 128)
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        curvature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ball = PoincareBall(c=curvature)
        self.n_layers = n_layers

        # Learnable node embedding (dipelajari dari struktur graph)
        self.node_emb = nn.Embedding(num_nodes, in_dim)
        nn.init.normal_(self.node_emb.weight, std=0.01)

        # Stack HGCN layers
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            HGCNLayer(dims[i], dims[i + 1], self.ball, dropout)
            for i in range(n_layers)
        ])

        # Layer norm di tangent space (setelah setiap layer)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(n_layers)
        ])

        self.out_dim = hidden_dim

    def forward(self, edge_index: Tensor) -> Tensor:
        """
        Forward pass: compute embeddings untuk semua node taxonomy.

        Parameter:
            edge_index : (2, E) — taxonomy tree edges (parent → child)

        Return:
            (N_nodes, hidden_dim) — node embeddings di Poincaré ball
        """
        N = self.node_emb.num_embeddings
        node_ids = torch.arange(N, device=edge_index.device)

        # Inisialisasi: learnable embedding → Poincaré ball via expmap0
        h = self.ball.expmap0(self.node_emb(node_ids))  # (N, in_dim)

        # Propagasi melalui HGCN layers
        for layer in self.layers:
            h_new = layer(h, edge_index)                 # (N, hidden_dim) di Poincaré

            # Tangent space
            h_tan = self.ball.logmap0(h_new)             # → tangent space

            # Hanya clamp jika norm meledak (> 10) — TIDAK meratakan semua norm.
            # LayerNorm dihapus karena ia memaksa semua node ke norm yang sama
            # (~sqrt(hidden_dim)), sehingga expmap0 selalu menghasilkan norm
            # identik (tanh(1) ≈ 0.7616) dan struktur kedalaman hierarki hilang.
            # Max-norm clamp mempertahankan variasi antar node sambil mencegah
            # ledakan gradien.
            norm = h_tan.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h_tan = h_tan * (10.0 / norm).clamp(max=1.0)
            h = self.ball.expmap0(h_tan)                 # → kembali ke Poincaré

        return h  # (N, hidden_dim) — di Poincaré ball


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY GRAPH BUILDER (dari label encoder ke edge_index)
# ─────────────────────────────────────────────────────────────────────────────

def build_taxonomy_graph(
    label_encoder,
    df=None,
    device: str = "cpu",
) -> Tuple[int, Tensor, dict, Tensor]:
    """
    Bangun taxonomy graph (edge_index) dari HierarchicalLabelEncoder.

    Jika df diberikan: bangun edges parent→child dari co-occurrence data nyata
    (setiap baris df menentukan parent-child relationship berdasarkan rank berurutan).
    Jika df=None: fallback ke koneksi sequential antar rank (backward compatible).

    Parameter:
        label_encoder : instance HierarchicalLabelEncoder
        df            : DataFrame dengan kolom TAX_RANKS (opsional, dari Preprocess.py)
                        Kolom: "Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"
        device        : target device ("cpu", "cuda", "mps")

    Return:
        (num_nodes, edge_index, node2idx, leaf_node_ids) dimana:
          num_nodes     : total jumlah node dalam taxonomy graph
          edge_index    : tensor (2, E) di device yang diminta
          node2idx      : dict {rank:label → global_node_id}
          leaf_node_ids : tensor (L,) int64 — global ID node Species (leaf level)
    """
    import pickle, os

    ranks = label_encoder.ranks  # ["Phylum", "Class", ..., "Species"]

    # Buat mapping global node ID
    global_id: dict = {}
    node_counter = [0]

    def get_id(rank: str, label: str) -> int:
        key = f"{rank}:{label}"
        if key not in global_id:
            global_id[key] = node_counter[0]
            node_counter[0] += 1
        return global_id[key]

    # Root node
    root_id = get_id("root", "root")

    src_list, dst_list = [], []

    if df is not None:
        # ── Co-occurrence based parent-child edges ────────────────────────
        # Iterasi semua rank pair (r_parent, r_child) dan kumpulkan edge unik
        # dari observasi nyata di data
        all_ranks = ["Domain"] + ranks  # Domain juga diinclude jika ada di df
        # Filter hanya rank yang ada sebagai kolom di df
        available_ranks = [r for r in all_ranks if r in df.columns]

        for i in range(len(available_ranks) - 1):
            r_parent = available_ranks[i]
            r_child  = available_ranks[i + 1]

            # Kumpulkan pasangan parent-child unik dari data
            pairs = (
                df[[r_parent, r_child]]
                .dropna()
                .apply(lambda col: col.astype(str).str.strip())
            )
            pairs = pairs[(pairs[r_parent] != "") & (pairs[r_child] != "")]
            unique_pairs = pairs.drop_duplicates()

            for _, row in unique_pairs.iterrows():
                pid = get_id(r_parent, row[r_parent])
                cid = get_id(r_child,  row[r_child])
                src_list.append(pid)
                dst_list.append(cid)

        # Sambungkan root → semua Phylum (rank pertama)
        if ranks:
            for lbl in label_encoder.label2idx[ranks[0]]:
                if lbl in ("<PAD>", "<UNK>"):
                    continue
                key = f"{ranks[0]}:{lbl}"
                if key in global_id:
                    src_list.append(root_id)
                    dst_list.append(global_id[key])

    else:
        # ── Fallback: sequential rank connections ─────────────────────────
        # Sambungkan root → setiap Phylum
        for lbl in label_encoder.label2idx[ranks[0]]:
            if lbl in ("<PAD>", "<UNK>"):
                continue
            child_id = get_id(ranks[0], lbl)
            src_list.append(root_id)
            dst_list.append(child_id)

        # Sambungkan antar rank berurutan: setiap parent → setiap child dalam ranknya
        # (simplified; lebih akurat jika df tersedia)
        for i in range(len(ranks) - 1):
            r_parent = ranks[i]
            r_child  = ranks[i + 1]
            for lbl_p in label_encoder.label2idx[r_parent]:
                if lbl_p in ("<PAD>", "<UNK>"):
                    continue
                pid = get_id(r_parent, lbl_p)
                for lbl_c in label_encoder.label2idx[r_child]:
                    if lbl_c in ("<PAD>", "<UNK>"):
                        continue
                    cid = get_id(r_child, lbl_c)
                    src_list.append(pid)
                    dst_list.append(cid)

    num_nodes = node_counter[0]

    # Kumpulkan leaf node IDs (Species level)
    species_rank = ranks[-1]  # "Species"
    leaf_ids = [
        global_id[f"{species_rank}:{lbl}"]
        for lbl in label_encoder.label2idx.get(species_rank, {})
        if lbl not in ("<PAD>", "<UNK>") and f"{species_rank}:{lbl}" in global_id
    ]
    leaf_node_ids = torch.tensor(leaf_ids, dtype=torch.long, device=device)

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long, device=device
    )

    return num_nodes, edge_index, global_id, leaf_node_ids


# ─────────────────────────────────────────────────────────────────────────────
# LCA UTILITY FOR ABSTENTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_lca(
    topk_leaf_ids: Tensor,
    node2idx: dict,
    label_encoder,
) -> dict:
    """
    Hitung Lowest Common Ancestor (LCA) dari top-k nearest leaf nodes.
    Digunakan saat OOD abstention: model tidak yakin → prediksi ke rank paling kasar
    yang masih konsisten di antara top-k Species kandidat.

    Parameter:
        topk_leaf_ids : (k,) int tensor — global node IDs dari Species terdekat
        node2idx      : dict {"rank:label" → global_id} dari build_taxonomy_graph()
        label_encoder : HierarchicalLabelEncoder

    Return:
        dict dengan key:
          "predictions" : {rank: label_string} — prediksi di setiap rank (kosong jika tidak konsisten)
          "lca_rank"    : rank terbawah yang masih konsisten ("Species", "Genus", ..., "unknown")
    """
    # Balik node2idx: global_id → "rank:label"
    idx2node = {v: k for k, v in node2idx.items()}

    ranks = label_encoder.ranks  # ["Phylum", ..., "Species"]
    species_rank = ranks[-1]

    # Petakan setiap leaf_id ke label Species-nya
    leaf_labels = []
    for gid in topk_leaf_ids.tolist():
        key = idx2node.get(gid, "")
        if key.startswith(f"{species_rank}:"):
            leaf_labels.append(key.split(":", 1)[1])
        else:
            leaf_labels.append("")

    if not any(leaf_labels):
        return {"predictions": {r: "" for r in ranks}, "lca_rank": "unknown"}

    # Untuk setiap leaf Species, cari semua ancestor di label_encoder
    # Kita cari ancestor berdasarkan label match di setiap rank
    # Karena kita tidak menyimpan full path per leaf, gunakan prefix matching dari node2idx

    # Bangun reverse map per rank: label → set of parent labels di rank di atasnya
    # Simplified: anggap leaf_labels valid, cari konsensus per rank
    predictions = {}
    lca_rank    = "unknown"

    for rank in reversed(ranks):  # Species → Genus → ... → Phylum
        # Kumpulkan semua label di rank ini yang berhubungan dengan topk leaves
        # Heuristik: cari node yang namanya prefix-match dengan salah satu Species
        rank_labels = []
        for sp_label in leaf_labels:
            if not sp_label:
                continue
            # Cari node di rank ini yang memiliki Species yang dimaksud sebagai keturunan
            # Gunakan node2idx: cari semua "{rank}:X" yang koeksist dengan species ini
            # Karena kita hanya punya co-occurrence data, lookup langsung dari label_encoder
            for node_key, gid in node2idx.items():
                if node_key.startswith(f"{rank}:") and gid in topk_leaf_ids.tolist():
                    rank_labels.append(node_key.split(":", 1)[1])
                    break
            else:
                # Fallback: ambil dari label_encoder idx2label langsung
                pass

        # Jika tidak dapat dari node2idx, coba species-rank lookup di node2idx transitif
        if not rank_labels:
            # Ambil semua "{rank}:X" node yang ada di node2idx, periksa apakah
            # mereka adalah ancestor dari setidaknya satu Species di topk
            for node_key in node2idx:
                if node_key.startswith(f"{rank}:"):
                    label = node_key.split(":", 1)[1]
                    rank_labels.append(label)
            # Simpan hanya yang paling sering (heuristic LCA)
            if rank_labels:
                from collections import Counter
                rank_labels = [Counter(rank_labels).most_common(1)[0][0]]

        unique_labels = set(rank_labels)
        if len(unique_labels) == 1 and unique_labels != {""} and unique_labels != {None}:
            predictions[rank] = unique_labels.pop()
            lca_rank = rank
        else:
            predictions[rank] = ""  # inkonsisten

    return {"predictions": predictions, "lca_rank": lca_rank}