#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_all_cv_curves_mdad_no_grid.py

修改说明（相对你上传的版本）：
1) 去掉“网格搜索/Trial sweeps”相关逻辑（hparam_grid_search 及其依赖函数与入口）。
2) 将所有输出（CSV/JSON/NPZ/PNG）的“存储路径”作为命令行参数传入：--export-dir。
3) 保留原有的 5-fold 训练与导出逻辑：fold_metrics.csv / roc_merged_curve.csv / pr_merged_curve.csv / summary.json / merged_labels_scores.npz，
   并在 export_dir 下额外输出 roc.png / pr.png 便于直接引用。

用法示例：
python main_all_cv_curves_mdad_no_grid.py --data-root MDAD --export-dir result/curves --neg-ratio 3.0
"""

import os
import json
import time
import csv
import argparse
import numpy as np
import torch




from mode_data_load import mode_data_load  # type: ignore


import ban3  # type: ignore
import rwr   # type: ignore


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    precision_recall_curve, roc_curve, auc
)

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
#                       通用工具
# ============================================================
def ensure_dir(p: str):
    """确保目录存在。"""
    os.makedirs(p, exist_ok=True)


def _parse_none_int(x: str):
    """命令行参数：支持 'None' 或整数。"""
    if x is None:
        return None
    if isinstance(x, str) and x.lower() == "none":
        return None
    return int(x)


def _parse_rf_max_features(x: str):
    """
    RandomForestClassifier(max_features=...) 支持：
      - "sqrt" / "log2"
      - float (0~1)：比例
      - int：特征数
    这里做一个尽量稳健的解析。
    """
    if x is None:
        return "sqrt"
    if isinstance(x, str):
        xs = x.strip()
        if xs.lower() in ("sqrt", "log2", "auto"):
            # sklearn 新版本中 "auto" 对分类器通常等价于 "sqrt"，保留原值以兼容历史
            return xs.lower()
        if xs.lower() == "none":
            return None
        # 先尝试 int
        try:
            return int(xs)
        except Exception:
            pass
        # 再尝试 float
        try:
            return float(xs)
        except Exception:
            pass
        return xs
    return x


def pick_threshold_by_f1(y_true, y_score):
    """
    在训练折上根据 PR 曲线挑选“F1 最大”的阈值。
    注意：precision_recall_curve 的 thresholds 数量比 p/r 少 1。
    """
    p, r, th = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1)
    return th[max(i - 1, 0)]


def tune_weight(y_true, p1, p2, metric="auc", grid=201):
    """
    在训练折上用网格搜索融合权重 w：
      fused = (1-w) * p1 + w * p2
    metric:
      - "auc"  : 优化 ROC-AUC
      - "aupr" : 优化 AUPR(AP)
    """
    ws = np.linspace(0.0, 1.0, grid)
    best_w, best_s = 0.5, -1.0
    for w in ws:
        pe = (1.0 - w) * p1 + w * p2
        if metric == "aupr":
            s = average_precision_score(y_true, pe)
        else:
            s = roc_auc_score(y_true, pe)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


# ============================================================
#                 图构建 / RWR / 特征抽取
# ============================================================
def gip_similarity(M, entity_axis=0, eps=1e-12):
    """GIP 相似度（矢量化实现）。"""
    X = M if entity_axis == 0 else M.T
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    gram = X @ X.T
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    gamma = X.shape[0] / (np.sum(sq) + eps)
    K = np.exp(-gamma * D2)
    return ((K + K.T) * 0.5).astype(np.float32)


def mask_h2_edges_for_test(te_pairs, A_mdis, A_ddis, side="microbe"):
    """
    对每个测试样本 (i_m, j_d)，屏蔽会形成两跳通路的疾病桥接边：
      - side="microbe": 仅屏蔽微生物→疾病边 (i_m, k)（推荐，最小破坏）
      - side="both"   : 同时屏蔽 (i_m, k) 和 (j_d, k)
    """
    A_mdis_train = A_mdis.copy().astype(np.float32)
    A_ddis_train = A_ddis.copy().astype(np.float32)

    m2dis = [np.where(A_mdis_train[i] > 0)[0] for i in range(A_mdis_train.shape[0])]
    d2dis = [np.where(A_ddis_train[j] > 0)[0] for j in range(A_ddis_train.shape[0])]

    for i_m, j_d in te_pairs:
        if i_m < 0 or j_d < 0:
            continue
        common = np.intersect1d(m2dis[i_m], d2dis[j_d], assume_unique=False)
        if common.size == 0:
            continue
        A_mdis_train[i_m, common] = 0.0
        if side == "both":
            A_ddis_train[j_d, common] = 0.0

    return A_mdis_train, A_ddis_train


def build_A_H2(A_mdis, A_ddis, n_m, n_dis, n_d):
    """构建 H2 异构图邻接矩阵（M-Dis-D）。"""
    n = n_m + n_dis + n_d
    A = np.zeros((n, n), dtype=np.float32)
    A[0:n_m, n_m:n_m + n_dis] = A_mdis
    A[n_m:n_m + n_dis, 0:n_m] = A_mdis.T
    A[n_m:n_m + n_dis, n_m + n_dis:] = A_ddis.T
    A[n_m + n_dis:, n_m:n_m + n_dis] = A_ddis
    return A


def build_H1_full(A_md, S_m, S_d, n_m, n_dis, n_d, lam_m=1.0, lam_d=1.0, eps_loop=1e-6):
    """
    构建论文 Eq.(9) 的结构增强异构邻接矩阵 H1（完整节点空间：M + Dis + D）。
      - lam_m：微生物相似度边权重系数（Eq.(9) 中的 λ_m）
      - lam_d：药物相似度边权重系数（Eq.(9) 中的 λ_d）
    """
    n = n_m + n_dis + n_d
    H = np.zeros((n, n), dtype=np.float32)

    lam_m = np.float32(lam_m)
    lam_d = np.float32(lam_d)

    H[0:n_m, 0:n_m] = lam_m * S_m
    H[n_m + n_dis:, n_m + n_dis:] = lam_d * S_d

    H[0:n_m, n_m + n_dis:] = A_md
    H[n_m + n_dis:, 0:n_m] = A_md.T

    # disease 节点极小自环，避免出现全零行
    if eps_loop > 0:
        i = np.arange(n_m, n_m + n_dis)
        H[i, i] = eps_loop

    return H


def row_normalize_with_selfloop(A):
    """行归一化：T = D^{-1}A；遇到全零行则补对角自环。"""
    A = np.maximum(A, 0.0).astype(np.float32)
    deg = A.sum(axis=1, keepdims=True)
    zero = (deg == 0)
    if zero.any():
        idx = np.where(zero.squeeze())[0]
        A[idx, idx] = 1.0
        deg = A.sum(axis=1, keepdims=True)
    return (A / deg).astype(np.float32)


def build_dis_context_Rdis(i_m, j_d, A_mdis, A_ddis, R_dis):
    """疾病上下文：取 microbe 或 drug 相连的疾病集合，在 R_dis 上做均值池化。"""
    dis_m = np.where(A_mdis[i_m] == 1)[0]
    dis_d = np.where(A_ddis[j_d] == 1)[0]
    idx = np.union1d(dis_m, dis_d)
    if idx.size == 0:
        return R_dis.mean(axis=0)
    return R_dis[idx].mean(axis=0)


def get_ban_probs(model, pairs, R_m, R_d, R_dis, A_mdis, A_ddis, device):
    """对给定 (m,d) 列表，用已训练 BAN 计算正类概率。"""
    model.eval()
    probs = []
    with torch.no_grad():
        for i_m, j_d in pairs:
            rm = torch.from_numpy(R_m[i_m]).float().unsqueeze(0).to(device)
            rd = torch.from_numpy(R_d[j_d]).float().unsqueeze(0).to(device)
            rdis_vec = build_dis_context_Rdis(i_m, j_d, A_mdis, A_ddis, R_dis).astype(np.float32)
            rdis = torch.from_numpy(rdis_vec).unsqueeze(0).to(device)
            logits, _ = model(rm, rd, rdis)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(p)
    return np.asarray(probs, dtype=np.float32)


def collect_z_mid(model, pairs, R_m, R_d, R_dis, A_mdis, A_ddis, device):
    """抽取 BAN 的中间表征 z_mid（作为 RF 的输入特征）。"""
    model.eval()
    zs = []
    with torch.no_grad():
        for i_m, j_d in pairs:
            rm = torch.from_numpy(R_m[i_m]).float().unsqueeze(0).to(device)
            rd = torch.from_numpy(R_d[j_d]).float().unsqueeze(0).to(device)
            rdis_vec = build_dis_context_Rdis(i_m, j_d, A_mdis, A_ddis, R_dis).astype(np.float32)
            rdis = torch.from_numpy(rdis_vec).unsqueeze(0).to(device)
            _, z = model(rm, rd, rdis)
            zs.append(z.squeeze(0).cpu().numpy())
    return np.stack(zs, axis=0).astype(np.float32)


def T_make(mode_data, A2, te_pairs, te_y, n_m, n_dis, n_d, lam=0.3, lam_m=1.0, lam_d=1.0):
    """
    构建训练折使用的转移矩阵 T：
      1) 屏蔽测试折中的正例边（防止信息泄漏）
      2) 基于训练图计算 GIP 相似度
      3) 构建 H2 与 Eq.(9) 的 H1_full 并融合
      4) 行归一化得到转移概率矩阵
    """
    # --- 训练图：屏蔽测试折正例边 ---
    A_md_train = (A2 >= 1).astype(np.float32)
    for (i_m, j_d), y in zip(te_pairs, te_y):
        if y == 1:
            A_md_train[i_m, j_d] = 0.0

    # --- 同类相似度（基于训练图） ---
    S_m = gip_similarity(A_md_train, entity_axis=0)
    S_d = gip_similarity(A_md_train, entity_axis=1)

    # --- 构建 H2（并屏蔽两跳泄漏） ---
    A_mdis_train, A_ddis_train = mask_h2_edges_for_test(
        te_pairs, mode_data.A_mdis, mode_data.A_ddis, side="microbe"
    )
    A_H2 = build_A_H2(A_mdis_train, A_ddis_train, n_m, n_dis, n_d)

    # --- 构建 H1_full（Eq.(9)） ---
    H1_full = build_H1_full(
        A_md_train, S_m, S_d, n_m, n_dis, n_d,
        lam_m=lam_m, lam_d=lam_d, eps_loop=1e-6
    )

    # --- 融合并归一化 ---
    A_all = (1.0 - lam) * A_H2 + lam * H1_full
    T = row_normalize_with_selfloop(A_all)
    return T


def rwr_handle(T, n_m, n_dis, n_d, alpha=0.5):
    """在统一图上做 RWR，得到各节点类型的稳态分布。"""
    N = n_m + n_dis + n_d
    idx_m = np.arange(0, n_m, dtype=int)
    idx_dis = np.arange(n_m, n_m + n_dis, dtype=int)
    idx_d = np.arange(n_m + n_dis, N, dtype=int)
    R_m = rwr.rwr_matrix(T, idx_m, alpha=alpha)
    R_dis = rwr.rwr_matrix(T, idx_dis, alpha=alpha)
    R_d = rwr.rwr_matrix(T, idx_d, alpha=alpha)
    return R_m, R_dis, R_d


def get_pos_neg_2(mode_data):
    """正样本：A_2>=1；负样本候选：A_2==0（未标注）。"""
    A2 = mode_data.A_2.astype(np.float32)
    pos = np.argwhere(A2 >= 1)
    neg = np.argwhere(A2 == 0)
    return A2, pos, neg


# ============================================================
#                    CV 汇总与导出
# ============================================================
def compute_summary_from_records(roc_records, fold_rows, pr_grid=1000, roc_grid=1000):
    """
    注意：
      - merged AUC：对合并后的 (y_all, s_all) 计算 ROC，再算 AUC
      - merged AUPR：对合并后的 (y_all, s_all) 直接算 average_precision_score (AP)
      - PR 曲线导出：使用 precision_recall_curve 的原始阶梯点（不做插值平均）
    """
    y_all = np.concatenate([y for (y, _) in roc_records])
    s_all = np.concatenate([s for (_, s) in roc_records])

    # merged ROC（插值仅用于输出曲线点，不影响 AUC 数值）
    fpr, tpr, _ = roc_curve(y_all, s_all, pos_label=1)
    merged_auc = auc(fpr, tpr)
    mean_fpr = np.linspace(0, 1, roc_grid)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)

    # merged PR（raw 点 + AP 数值）
    precision_raw, recall_raw, _ = precision_recall_curve(y_all, s_all, pos_label=1)
    merged_aupr = average_precision_score(y_all, s_all)

    # per-fold stats
    mean_fpr2 = np.linspace(0, 1, roc_grid)
    tprs, aucs = [], []
    aps = []

    fold_accs = [row["acc"] for row in fold_rows if "acc" in row]
    fold_f1s = [row["f1"] for row in fold_rows if "f1" in row]

    for (y, s) in roc_records:
        if len(np.unique(y)) < 2:
            continue
        fpr_i, tpr_i, _ = roc_curve(y, s, pos_label=1)
        aucs.append(auc(fpr_i, tpr_i))
        tpr_interp_i = np.interp(mean_fpr2, fpr_i, tpr_i)
        tpr_interp_i[0] = 0.0
        tprs.append(tpr_interp_i)

        aps.append(average_precision_score(y, s))

    summary = {
        "merged_auc": float(merged_auc),
        "merged_aupr": float(merged_aupr),
        "pos_ratio": float(y_all.mean()),
        "n_samples": int(y_all.size),

        "fold_mean_auc": float(np.mean(aucs)) if aucs else None,
        "fold_std_auc": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
        "fold_mean_aupr": float(np.mean(aps)) if aps else None,
        "fold_std_aupr": float(np.std(aps)) if len(aps) > 1 else 0.0,
        "fold_mean_acc": float(np.mean(fold_accs)) if fold_accs else None,
        "fold_std_acc": float(np.std(fold_accs)) if len(fold_accs) > 1 else 0.0,
        "fold_mean_f1": float(np.mean(fold_f1s)) if fold_f1s else None,
        "fold_std_f1": float(np.std(fold_f1s)) if len(fold_f1s) > 1 else 0.0,

        # ROC 可视化点（均值曲线）
        "mean_fpr": mean_fpr2.tolist(),
        "mean_tpr": (np.mean(tprs, axis=0).tolist() if tprs else None),

        # PR 不输出 mean 曲线，避免插值引入的形状偏差
        "mean_recall": None,
        "mean_precision": None,
    }
    return summary, (mean_fpr, tpr_interp), (recall_raw, precision_raw)


def export_cv_values(roc_records, fold_rows, outdir="result/curves", pr_grid=1000, roc_grid=1000):
    """将 CV 产物写到 outdir。"""
    ensure_dir(outdir)

    # fold metrics
    fold_csv = os.path.join(outdir, "fold_metrics.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        for row in fold_rows:
            writer.writerow(row)

    summary, (mean_fpr, tpr_interp), (recall_raw, precision_raw) = compute_summary_from_records(
        roc_records, fold_rows, pr_grid=pr_grid, roc_grid=roc_grid
    )

    # ROC curve points
    roc_merged_csv = os.path.join(outdir, "roc_merged_curve.csv")
    np.savetxt(
        roc_merged_csv,
        np.c_[mean_fpr, tpr_interp],
        delimiter=",",
        header="fpr,tpr",
        comments="",
    )

    # PR curve points（raw）
    pr_merged_csv = os.path.join(outdir, "pr_merged_curve.csv")
    np.savetxt(
        pr_merged_csv,
        np.c_[recall_raw, precision_raw],
        delimiter=",",
        header="recall,precision",
        comments="",
    )

    # ROC mean curve（便于做阴影带；这里 std 置 0 兼容旧绘图逻辑）
    if summary.get("mean_tpr") is not None:
        roc_mean_csv = os.path.join(outdir, "roc_mean_curve.csv")
        std_dummy = np.zeros_like(summary["mean_fpr"], dtype=float)
        np.savetxt(
            roc_mean_csv,
            np.c_[np.asarray(summary["mean_fpr"]), np.asarray(summary["mean_tpr"]), std_dummy],
            delimiter=",",
            header="fpr,mean_tpr,std_tpr",
            comments="",
        )

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存 raw label/score（供日后复算/画图）
    y_all = np.concatenate([y for (y, _) in roc_records])
    s_all = np.concatenate([s for (_, s) in roc_records])
    np.savez_compressed(
        os.path.join(outdir, "merged_labels_scores.npz"),
        y_all=y_all.astype(np.int64),
        s_all=s_all.astype(np.float32),
    )

    print(f"[EXPORT] CV curves saved to: {outdir}")
    print(f"        merged AUC={summary['merged_auc']:.6f}, merged AUPR={summary['merged_aupr']:.6f}")
    return summary


def save_roc_fig(curves_dir, out_png):
    """根据 roc_merged_curve.csv（与可选的 roc_mean_curve.csv）画 ROC 图。"""
    arr = np.loadtxt(os.path.join(curves_dir, "roc_merged_curve.csv"), delimiter=",", skiprows=1)
    fpr, tpr = arr[:, 0], arr[:, 1]
    mean_csv = os.path.join(curves_dir, "roc_mean_curve.csv")
    m_fpr = m_tpr = m_std = None
    if os.path.exists(mean_csv):
        arrm = np.loadtxt(mean_csv, delimiter=",", skiprows=1)
        m_fpr, m_tpr, m_std = arrm[:, 0], arrm[:, 1], arrm[:, 2]

    title = "ROC"
    s_path = os.path.join(curves_dir, "summary.json")
    if os.path.exists(s_path):
        with open(s_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            if "merged_auc" in d:
                try:
                    title = f"ROC (merged AUC={float(d['merged_auc']):.4f})"
                except Exception:
                    pass

    plt.figure()
    plt.plot(fpr, tpr, label="Merged ROC")
    if m_fpr is not None:
        plt.plot(m_fpr, m_tpr, label="Mean ROC")
        if m_std is not None:
            plt.fill_between(m_fpr, m_tpr - m_std, m_tpr + m_std, alpha=0.2, label="±1 std")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png


def save_pr_fig(curves_dir, out_png):
    """根据 pr_merged_curve.csv（raw PR 点）画 PR 图。"""
    arr = np.loadtxt(os.path.join(curves_dir, "pr_merged_curve.csv"), delimiter=",", skiprows=1)
    recall, precision = arr[:, 0], arr[:, 1]

    title = "PR"
    s_path = os.path.join(curves_dir, "summary.json")
    if os.path.exists(s_path):
        with open(s_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            if "merged_aupr" in d:
                try:
                    title = f"PR (merged AUPR={float(d['merged_aupr']):.4f})"
                except Exception:
                    pass

    plt.figure()
    plt.step(recall, precision, where="post", label="Merged PR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png


# ============================================================
#                   主流程：5-fold 训练与评估
# ============================================================
def fold_run(
    mode_data,
    kf_seed=20250825,
    # ===== 训练相关超参数（BAN） =====
    hidden_dim=64,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    batch_size=256,
    # ===== RF 相关超参数 =====
    rf_n_estimators=600,
    rf_max_depth=None,
    rf_max_features="sqrt",
    # ===== 融合与阈值相关超参数 =====
    fuse_metric="aupr",
    fuse_grid=201,
    w_h=0.0,
    # ===== 图构建/RWR 相关超参数 =====
    lam=0.3,
    lam_m=1.0,
    lam_d=1.0,
    rwr_alpha=0.5,
    # ===== 采样相关超参数 =====
    neg_ratio=1.0,
    # ===== 输出路径（关键改动：作为参数传入） =====
    export_dir="result/curves",
):
    """
    进行 Stratified 5-fold CV，并输出：
      - fold_metrics.csv
      - roc_merged_curve.csv / roc_mean_curve.csv
      - pr_merged_curve.csv
      - summary.json
      - merged_labels_scores.npz
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    n_m, n_dis, n_d = mode_data.n_m, mode_data.n_dis, mode_data.n_d

    A2, pos, neg = get_pos_neg_2(mode_data)

    # -------- 负采样：从未观测(0)配对中无放回随机抽取伪负样本 --------
    rng = np.random.default_rng(kf_seed)
    rng.shuffle(neg)
    neg_keep = int(np.floor(len(pos) * float(neg_ratio)))
    neg_keep = max(1, neg_keep)
    neg_keep = min(len(neg), neg_keep)
    neg = neg[:neg_keep]  # 正:负 = 1:neg_ratio

    samples = np.vstack([pos, neg]).astype(np.int64)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=kf_seed)

    roc_records = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(samples, labels), 1):
        tr_pairs, te_pairs = samples[train_idx], samples[test_idx]
        tr_y, te_y = labels[train_idx], labels[test_idx]

        # 1) 构图 + RWR（严格只使用训练信息）
        T = T_make(mode_data, A2, te_pairs, te_y, n_m, n_dis, n_d, lam=lam, lam_m=lam_m, lam_d=lam_d)
        R_m, R_dis, R_d = rwr_handle(T, n_m, n_dis, n_d, alpha=rwr_alpha)

        # 2) 训练 BAN
        model = ban3.BANModel(in_dim=(n_m + n_dis + n_d), hidden_dim=hidden_dim, out_dim=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        ce = torch.nn.CrossEntropyLoss()

        model.train()
        tr_idx_arr = np.arange(len(tr_pairs))
        for _ in range(epochs):
            np.random.shuffle(tr_idx_arr)
            for s in range(0, len(tr_idx_arr), batch_size):
                batch_idx = tr_idx_arr[s:s + batch_size]
                ms = tr_pairs[batch_idx, 0]
                ds = tr_pairs[batch_idx, 1]
                yb = torch.from_numpy(tr_y[batch_idx]).long().to(device)

                rm = torch.from_numpy(R_m[ms]).float().to(device)
                rd = torch.from_numpy(R_d[ds]).float().to(device)
                rdis_np = np.vstack([
                    build_dis_context_Rdis(i, j, mode_data.A_mdis, mode_data.A_ddis, R_dis)
                    for i, j in zip(ms, ds)
                ]).astype(np.float32)
                rdis = torch.from_numpy(rdis_np).to(device)

                logits, _ = model(rm, rd, rdis)
                loss = ce(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # 3) BAN/RF 概率 + 融合
        ban_probs_tr = get_ban_probs(model, tr_pairs, R_m, R_d, R_dis, mode_data.A_mdis, mode_data.A_ddis, device)
        ban_probs_te = get_ban_probs(model, te_pairs, R_m, R_d, R_dis, mode_data.A_mdis, mode_data.A_ddis, device)

        Z_tr = collect_z_mid(model, tr_pairs, R_m, R_d, R_dis, mode_data.A_mdis, mode_data.A_ddis, device)
        Z_te = collect_z_mid(model, te_pairs, R_m, R_d, R_dis, mode_data.A_mdis, mode_data.A_ddis, device)

        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            max_features=rf_max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=kf_seed,
        )
        rf.fit(Z_tr, tr_y)
        rf_probs_tr = rf.predict_proba(Z_tr)[:, 1]
        rf_probs_te = rf.predict_proba(Z_te)[:, 1]

        # 训练折上自动选择融合权重 w（或用 w_h 覆盖）
        w, _ = tune_weight(tr_y, ban_probs_tr, rf_probs_tr, metric=fuse_metric, grid=fuse_grid)
        if float(w_h) != 0.0:
            w = float(w_h)

        fused_tr = (1.0 - w) * ban_probs_tr + w * rf_probs_tr
        thr = pick_threshold_by_f1(tr_y, fused_tr)

        fused_te = (1.0 - w) * ban_probs_te + w * rf_probs_te
        pred_te = (fused_te >= thr).astype(np.int64)

        # 4) 评价指标（AUC/AUPR 用连续分数；ACC/F1 用阈值二值化）
        auc_val = roc_auc_score(te_y, fused_te)
        aupr_val = average_precision_score(te_y, fused_te)
        acc = accuracy_score(te_y, pred_te)
        f1 = f1_score(te_y, pred_te)

        roc_records.append((te_y.copy(), fused_te.copy()))
        fold_rows.append({
            "fold": fold,
            "n_test": int(te_y.size),
            "auc": float(auc_val),
            "aupr": float(aupr_val),
            "acc": float(acc),
            "f1": float(f1),
            "w": float(w),
            "thr": float(thr),
        })

        print(f"[Fold {fold}] AUC={auc_val:.6f} AUPR={aupr_val:.6f} ACC={acc:.6f} F1={f1:.6f} w={w:.4f} thr={thr:.6f}")

    # 输出
    return export_cv_values(roc_records, fold_rows, outdir=export_dir)


# ============================================================
#                          CLI 入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Run 5-fold CV for BAN+RF (no grid search).")
    ap.add_argument("--data-root", default="MDAD", help="数据目录（mode_data_load 的输入）")
    ap.add_argument("--export-dir", default="result/curves", help="输出目录（CSV/JSON/NPZ/PNG 全部写入这里）")

    # 采样/随机种子
    ap.add_argument("--seed", type=int, default=20250825, help="随机种子（同时用于负采样与 StratifiedKFold）")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="负采样比例：负/正，例如 3.0 表示 1:3")

    # BAN
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)

    # RF
    ap.add_argument("--rf-n-estimators", type=int, default=600)
    ap.add_argument("--rf-max-depth", type=_parse_none_int, default=None)
    ap.add_argument("--rf-max-features", type=str, default="sqrt")

    # 融合
    ap.add_argument("--fuse-metric", choices=["auc", "aupr"], default="aupr")
    ap.add_argument("--fuse-grid", type=int, default=201)
    ap.add_argument("--w-h", type=float, default=0.0, help="手动固定融合权重 w（0 表示自动搜索）")

    # 图构建/RWR（含 Eq.(9)）
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--lam-m", type=float, default=1.0)
    ap.add_argument("--lam-d", type=float, default=1.0)
    ap.add_argument("--rwr-alpha", type=float, default=0.5)

    args = ap.parse_args()


    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(script_dir, args.data_root)
    if not os.path.isabs(args.export_dir):
        args.export_dir = os.path.join(script_dir, args.export_dir)
    os.makedirs(args.export_dir, exist_ok=True)

    rf_max_features = _parse_rf_max_features(args.rf_max_features)

    print("[INFO] Loading data ...")
    mode_data = mode_data_load(args.data_root)

    ensure_dir(args.export_dir)
    t0 = time.time()

    summary = fold_run(
        mode_data,
        kf_seed=args.seed,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_max_features=rf_max_features,
        fuse_metric=args.fuse_metric,
        fuse_grid=args.fuse_grid,
        w_h=args.w_h,
        lam=args.lam,
        lam_m=args.lam_m,
        lam_d=args.lam_d,
        rwr_alpha=args.rwr_alpha,
        neg_ratio=args.neg_ratio,
        export_dir=args.export_dir,
    )

    # 额外输出 ROC/PR PNG（便于论文/回复直接引用）
    roc_png = os.path.join(args.export_dir, "roc.png")
    pr_png = os.path.join(args.export_dir, "pr.png")
    try:
        save_roc_fig(args.export_dir, roc_png)
        save_pr_fig(args.export_dir, pr_png)
        print(f"[FIG] ROC -> {roc_png}")
        print(f"[FIG] PR  -> {pr_png}")
    except Exception as e:
        print("[WARN] plotting failed:", e)

    elapsed = time.time() - t0
    print(f"[DONE] elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
