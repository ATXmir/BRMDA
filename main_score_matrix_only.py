# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier

from net_code.mode_data.mode_data_load import mode_data_load
from net_code.mode_ban_rf import ban3, rwr


# ----------------------------
# 基础工具函数
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def gip_similarity(M, entity_axis=0, eps=1e-12):
    """
    Gaussian Interaction Profile (GIP) similarity.
    entity_axis=0：按行实体（微生物）计算；entity_axis=1：按列实体（药物）计算。
    """
    X = M if entity_axis == 0 else M.T
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    gram = X @ X.T
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    gamma = X.shape[0] / (np.sum(sq) + eps)
    K = np.exp(-gamma * D2)
    return ((K + K.T) * 0.5).astype(np.float32)


def build_A_H2(A_mdis, A_ddis, n_m, n_dis, n_d):
    """
    构建 H2 异构图邻接矩阵（microbe-disease-drug）。
    A_mdis: [n_m, n_dis]
    A_ddis: [n_d, n_dis]
    返回 A: [n_m+n_dis+n_d, n_m+n_dis+n_d]
    """
    n = n_m + n_dis + n_d
    A = np.zeros((n, n), dtype=np.float32)
    A[0:n_m, n_m:n_m+n_dis] = A_mdis
    A[n_m:n_m+n_dis, 0:n_m] = A_mdis.T
    A[n_m:n_m+n_dis, n_m+n_dis:] = A_ddis.T
    A[n_m+n_dis:, n_m:n_m+n_dis] = A_ddis
    return A


def build_H1_full(A_md, S_m, S_d, n_m, n_dis, n_d, eps_loop=1e-6):
    """
    构建 H1（microbe-drug）增强图的扩展邻接矩阵，并在 disease 节点加极小自环避免孤点。
    A_md: [n_m, n_d] microbe-drug incidence
    S_m : [n_m, n_m] microbe similarity (GIP/融合后亦可)
    S_d : [n_d, n_d] drug similarity
    """
    n = n_m + n_dis + n_d
    H = np.zeros((n, n), dtype=np.float32)
    H[0:n_m, 0:n_m] = S_m
    H[n_m+n_dis:, n_m+n_dis:] = S_d
    H[0:n_m, n_m+n_dis:] = A_md
    H[n_m+n_dis:, 0:n_m] = A_md.T
    # disease 自环（数值很小，仅用于保证可达性/可归一化）
    if eps_loop > 0:
        i = np.arange(n_m, n_m+n_dis)
        H[i, i] = eps_loop
    return H


def row_normalize_with_selfloop(A):
    """
    行归一化：T = D^{-1} A
    若出现全零行，则给该行对应对角线补 1（自环），避免除零。
    """
    A = np.maximum(A, 0.0).astype(np.float32)
    deg = A.sum(axis=1, keepdims=True)
    zero = (deg == 0)
    if zero.any():
        idx = np.where(zero.squeeze())[0]
        A[idx, idx] = 1.0
        deg = A.sum(axis=1, keepdims=True)
    return (A / deg).astype(np.float32)


def build_transition_matrix_full(mode_data, A2, lam=0.3):
    """
    构建统一图的转移概率矩阵 T（用于 RWR）。
    """
    n_m, n_dis, n_d = mode_data.n_m, mode_data.n_dis, mode_data.n_d

    A_md = (A2 >= 1).astype(np.float32)
    S_m = gip_similarity(A_md, entity_axis=0)
    S_d = gip_similarity(A_md, entity_axis=1)

    A_H2 = build_A_H2(mode_data.A_mdis, mode_data.A_ddis, n_m, n_dis, n_d)
    H1_full = build_H1_full(A_md, S_m, S_d, n_m, n_dis, n_d, eps_loop=1e-6)

    A_all = (1.0 - lam) * A_H2 + lam * H1_full
    T = row_normalize_with_selfloop(A_all)
    return T


def rwr_handle(T, n_m, n_dis, n_d, alpha=0.5):
    """
    对统一图做 RWR，分别得到 microbe/drug/disease 的稳态分布（结构嵌入）。
    """
    N = n_m + n_dis + n_d
    idx_m = np.arange(0, n_m, dtype=int)
    idx_dis = np.arange(n_m, n_m + n_dis, dtype=int)
    idx_d = np.arange(n_m + n_dis, N, dtype=int)
    R_m = rwr.rwr_matrix(T, idx_m, alpha=alpha)
    R_dis = rwr.rwr_matrix(T, idx_dis, alpha=alpha)
    R_d = rwr.rwr_matrix(T, idx_d, alpha=alpha)
    return R_m, R_dis, R_d


def get_pos_neg_from_A2(mode_data, neg_ratio=1.0, seed=20250825):
    """
    从 A_2 中构造训练样本：
    - 正样本：A_2 >= 1
    - 负样本：从 A_2 == 0 中随机抽取 (neg_ratio * n_pos)
    """
    rng = np.random.default_rng(seed)
    A2 = mode_data.A_2.astype(np.float32)
    pos = np.argwhere(A2 >= 1)
    neg = np.argwhere(A2 == 0)

    rng.shuffle(neg)
    n_neg = int(len(pos) * float(neg_ratio))
    n_neg = min(n_neg, len(neg))
    neg = neg[:n_neg]

    samples = np.vstack([pos, neg]).astype(np.int64)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return A2, samples, labels


def build_dis_context_Rdis(i_m, j_d, A_mdis, A_ddis, R_dis):
    """
    disease 上下文聚合：
    对于给定 (microbe i_m, drug j_d)，取其关联 disease 的并集，
    然后在 R_dis 上做均值池化作为第三模态输入。
    """
    dis_m = np.where(A_mdis[i_m] == 1)[0]
    dis_d = np.where(A_ddis[j_d] == 1)[0]
    idx = np.union1d(dis_m, dis_d)
    if idx.size == 0:
        return R_dis.mean(axis=0)
    return R_dis[idx].mean(axis=0)


def infer_probs_and_z(model, pairs, R_m, R_d, R_dis, A_mdis, A_ddis, device, batch_size=2048):
    """
    对 pairs（shape: [N,2]）批量推理：
    - probs: BAN 的 softmax 概率（正类）
    - z    : BAN 的中间表示（用于 RF）
    """
    model.eval()
    N = pairs.shape[0]
    probs = np.empty((N,), dtype=np.float32)
    zs = []

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            batch = pairs[s:e]
            ms = batch[:, 0]
            ds = batch[:, 1]

            rm = torch.from_numpy(R_m[ms]).float().to(device)
            rd = torch.from_numpy(R_d[ds]).float().to(device)

            # rdis 仍需逐样本聚合（依赖 A_mdis/A_ddis 的稀疏关系）
            rdis_np = np.vstack([
                build_dis_context_Rdis(i, j, A_mdis, A_ddis, R_dis)
                for i, j in zip(ms, ds)
            ]).astype(np.float32)
            rdis = torch.from_numpy(rdis_np).float().to(device)

            logits, z = model(rm, rd, rdis)  # logits: [B,2], z: [B,hidden]
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
            probs[s:e] = p
            zs.append(z.detach().cpu().numpy().astype(np.float32))

    Z = np.concatenate(zs, axis=0)
    return probs, Z


# ----------------------------
# 训练 + 生成全量得分矩阵
# ----------------------------
def train_and_score_matrix(
    mode_data,
    out_dir="result",
    # ---- BAN ----
    hidden_dim=64,
    lr=5e-3,
    weight_decay=1e-4,
    epochs=15,
    batch_size=128,
    # ---- RF ----
    rf_n_estimators=300,
    rf_max_depth=None,
    rf_max_features=0.01,
    # ---- Graph/RWR ----
    lam=0.3,
    rwr_alpha=0.4,
    # ---- Sampling / Fusion ----
    neg_ratio=1.0,
    seed=20250825,
    w_fuse=0.5,
    infer_batch_size=2048,
):
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    n_m, n_dis, n_d = mode_data.n_m, mode_data.n_dis, mode_data.n_d

    A2, train_pairs, train_y = get_pos_neg_from_A2(mode_data, neg_ratio=neg_ratio, seed=seed)
    print(f"[INFO] Train samples: {train_pairs.shape[0]} (pos={int(train_y.sum())}, neg={int((train_y==0).sum())})")

    print("[INFO] Building transition matrix T and running RWR ...")
    t0 = time.time()
    T = build_transition_matrix_full(mode_data, A2, lam=lam)
    R_m, R_dis, R_d = rwr_handle(T, n_m, n_dis, n_d, alpha=rwr_alpha)
    print(f"[INFO] RWR done. elapsed={time.time()-t0:.2f}s")

    print("[INFO] Training BAN ...")
    model = ban3.BANModel(in_dim=(n_m + n_dis + n_d), hidden_dim=hidden_dim, out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()

    idx = np.arange(train_pairs.shape[0])
    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        model.train()
        ep_loss = 0.0
        n_seen = 0

        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            pairs_b = train_pairs[b]
            yb = torch.from_numpy(train_y[b]).long().to(device)

            ms = pairs_b[:, 0]
            ds = pairs_b[:, 1]

            rm = torch.from_numpy(R_m[ms]).float().to(device)
            rd = torch.from_numpy(R_d[ds]).float().to(device)
            rdis_np = np.vstack([
                build_dis_context_Rdis(i, j, mode_data.A_mdis, mode_data.A_ddis, R_dis)
                for i, j in zip(ms, ds)
            ]).astype(np.float32)
            rdis = torch.from_numpy(rdis_np).float().to(device)

            logits, _ = model(rm, rd, rdis)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.item()) * len(b)
            n_seen += len(b)

        print(f"  [BAN] epoch {ep:02d}/{epochs}  loss={ep_loss/max(n_seen,1):.6f}")

    print("[INFO] Extracting Z for RF training ...")
    _, Z_tr = infer_probs_and_z(
        model, train_pairs, R_m, R_d, R_dis,
        mode_data.A_mdis, mode_data.A_ddis, device, batch_size=infer_batch_size
    )

    print("[INFO] Training RF ...")
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        max_features=rf_max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(Z_tr, train_y)

    print("[INFO] Predicting full score matrix ...")
    ms_all = np.repeat(np.arange(n_m, dtype=np.int64), n_d)
    ds_all = np.tile(np.arange(n_d, dtype=np.int64), n_m)
    all_pairs = np.stack([ms_all, ds_all], axis=1)

    ban_probs_all, Z_all = infer_probs_and_z(
        model, all_pairs, R_m, R_d, R_dis,
        mode_data.A_mdis, mode_data.A_ddis, device, batch_size=infer_batch_size
    )
    rf_probs_all = rf.predict_proba(Z_all)[:, 1].astype(np.float32)

    w = float(w_fuse)
    fused = (1.0 - w) * ban_probs_all + w * rf_probs_all
    score_matrix = fused.reshape(n_m, n_d).astype(np.float32)

    npy_path = os.path.join(out_dir, "score_matrix.npy")
    csv_path = os.path.join(out_dir, "score_matrix.csv")
    cfg_path = os.path.join(out_dir, "run_config.json")

    np.save(npy_path, score_matrix)

    np.savetxt(csv_path, score_matrix, delimiter=",", fmt="%.6g")

    cfg = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_m": int(n_m), "n_d": int(n_d), "n_dis": int(n_dis),
        "train": {
            "neg_ratio": float(neg_ratio),
            "seed": int(seed),
            "n_train": int(train_pairs.shape[0]),
            "n_pos": int(train_y.sum()),
            "n_neg": int((train_y == 0).sum()),
        },
        "ban": {
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
        },
        "rf": {
            "n_estimators": int(rf_n_estimators),
            "max_depth": rf_max_depth,
            "max_features": rf_max_features,
        },
        "graph": {
            "lam": float(lam),
            "rwr_alpha": float(rwr_alpha),
        },
        "fusion": {
            "w_fuse": float(w),
            "formula": "score = (1-w)*BAN + w*RF",
        },
        "outputs": {
            "score_matrix_npy": os.path.abspath(npy_path),
            "score_matrix_csv": os.path.abspath(csv_path),
        }
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("[DONE] Score matrix saved:")
    print(f"  - {npy_path}")
    print(f"  - {csv_path}")
    print(f"  - {cfg_path}")

    return score_matrix


# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    mode_data = mode_data_load("data/source")

    train_and_score_matrix(
        mode_data,
        out_dir="result",
        hidden_dim=64,
        lr=0.005,
        weight_decay=1e-4,
        rf_n_estimators=300,
        rf_max_features=0.01,
        lam=0.3,
        rwr_alpha=0.4,
        epochs=15,
        batch_size=128,
        neg_ratio=1.0,
        seed=20250825,
        w_fuse=0.5,          
        infer_batch_size=2048
    )
