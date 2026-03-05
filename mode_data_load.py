# -*- coding: utf-8 -*-
"""mode_data_load_mdad.py

目标：
  - 读取 MDAD 数据文件，构建 A_mdis / A_ddis / A_dmic，以及 A_H2 / A_2。
  - middle 输出路径可配置（--middle-dir），也可关闭写盘（--no-middle-save）。

期望的数据文件（位于 data-root 目录下）：
  - microbes.xlsx
  - disease.xlsx
  - drugs.xlsx
  - microbe_with_disease.txt
  - drug_with_disease.txt
  - drug_microbe_matrix.txt



"""

import os
from typing import Optional

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import savemat


def gip_similarity(M, entity_axis=0, eps=1e-12):
    """GIP 高斯相似度。

    entity_axis:
      - 0：按行实体（微生物）计算
      - 1：按列实体（药物）计算
    """
    X = M if entity_axis == 0 else M.T
    X = np.asarray(X, dtype=np.float64)

    sq = np.sum(X * X, axis=1)
    gram = X @ X.T
    D2 = sq[:, None] + sq[None, :] - 2.0 * gram
    D2 = np.maximum(D2, 0.0)

    gamma = X.shape[0] / (np.sum(sq) + eps)
    K = np.exp(-gamma * D2)
    K = (K + K.T) * 0.5
    K = np.clip(K, 0.0, 1.0)
    return K


def _safe_int_or_str(x):
    """将 x 转换为 int（优先），失败则转 str。"""
    try:
        # pandas 可能读出 float，需要先转 int
        if isinstance(x, float) and (abs(x - int(x)) < 1e-12):
            return int(x)
        return int(x)
    except Exception:
        return str(x).strip()


def _read_excel_id_map(xlsx_path: str, candidate_cols):
    """读取 Excel 并返回“ID -> 1-based index”的映射表。"""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Excel 文件不存在：{xlsx_path}")

    # 优先使用 openpyxl；若环境缺失，会抛异常
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    except Exception:
        # 尝试让 pandas 自选引擎（仍可能需要 openpyxl）
        df = pd.read_excel(xlsx_path, sheet_name=0, engine=None)

    if df.shape[1] == 0:
        raise RuntimeError(f"Excel 为空：{xlsx_path}")

    col = None
    for c in candidate_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]  # fallback

    id_map = {}
    for idx, v in enumerate(df[col].tolist()):
        key = _safe_int_or_str(v)
        # 1-based index，保持与原始代码一致
        id_map[key] = idx + 1

    if len(id_map) == 0:
        raise RuntimeError(f"无法从 {xlsx_path} 读取任何 ID（列={col}）")

    return id_map


def _validate_required_files(data_root: str):
    required = [
        "microbes.xlsx",
        "disease.xlsx",
        "drugs.xlsx",
        "microbe_with_disease.txt",
        "drug_with_disease.txt",
        "drug_microbe_matrix.txt",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(data_root, f))]
    if missing:
        listed = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            f"数据目录缺少必要文件：{data_root}\n缺失文件：{listed}\n"
            f"请确认这些文件都位于同一目录下。"
        )


class mode_data_load:
    """MDAD 数据加载与异构图构建。"""

    def __init__(self, path: str, middle_dir: Optional[str] = None, save_middle: bool = True):
        # 统一转绝对路径，避免工作目录变化导致找不到文件
        self.data_root = os.path.abspath(path)
        print("[INFO] data_root:", self.data_root)

        _validate_required_files(self.data_root)

        # 1) 加载索引表 + A_dmic
        self._load_file_data(self.data_root)

        # 2) 构建 A_mdis / A_ddis / A_dmic
        A_mdis, A_ddis, A_dmic = self._build_adjacency_matrices(self.data_root)

        # 3) middle 输出路径（可配置）
        if middle_dir is None:
            # 默认放在 data_root 的上级目录：<project_root>/middle
            self.save_data_middle_path = os.path.abspath(os.path.join(self.data_root, "..", "middle"))
        else:
            self.save_data_middle_path = os.path.abspath(middle_dir)

        if save_middle:
            os.makedirs(self.save_data_middle_path, exist_ok=True)
            savemat(os.path.join(self.save_data_middle_path, "A_mdis.mat"), {"A_mdis": np.array(A_mdis)})
            savemat(os.path.join(self.save_data_middle_path, "A_ddis.mat"), {"A_ddis": np.array(A_ddis)})
            savemat(os.path.join(self.save_data_middle_path, "A_dmic.mat"), {"A_dmic": np.array(A_dmic)})

        # 4) 构建 A_H2 / A_2
        self._build_AH2_and_A2(A_mdis, A_ddis)
        if save_middle:
            savemat(os.path.join(self.save_data_middle_path, "example.mat"), {"example_matrix": np.array(self.A_2)})

        # 5) 同时构建一个融合后的转移矩阵 T（可选，主流程也可自行构建）
        #    这里保持与原逻辑一致：在 A_dmic 上算 GIP 相似度，构 H1，再与 H2 融合
        S_m = gip_similarity(self.A_dmic, entity_axis=0)
        S_d = gip_similarity(self.A_dmic, entity_axis=1)
        A_H1 = self._build_A_H1(S_m, self.A_dmic, S_d)
        A = self._merge_H1_H2(A_H1, self.A_H2, lambda_factor=0.1)
        A = np.maximum(A, 0.0)

        # 给零度节点自环，避免除零
        deg = A.sum(axis=1, keepdims=True)
        zero = (deg == 0)
        if zero.any():
            idx = np.where(zero.squeeze())[0]
            A[idx, idx] = 1.0

        deg = A.sum(axis=1, keepdims=True)
        self.T = (A / deg).astype(np.float32)

    # -----------------------------
    # 文件加载
    # -----------------------------
    def _load_file_data(self, path: str):
        self.microbes_table = _read_excel_id_map(os.path.join(path, "microbes.xlsx"), ["Microbe_ID", "microbe_id", "ID"])
        self.disease_table = _read_excel_id_map(os.path.join(path, "disease.xlsx"), ["Disease_ID", "disease_id", "Unnamed: 0", "ID"])
        self.drugs_table = _read_excel_id_map(os.path.join(path, "drugs.xlsx"), ["Drug_ID", "drug_id", "ID"])

        self.drug_dis_file = os.path.join(path, "drug_with_disease.txt")
        self.microbe_dis_file = os.path.join(path, "microbe_with_disease.txt")

        # 单独加载药物-微生物邻接矩阵（可能是 n_m×n_d 或 n_d×n_m）
        self.A_dmic = np.loadtxt(os.path.join(path, "drug_microbe_matrix.txt"))

    # -----------------------------
    # 邻接矩阵构建
    # -----------------------------
    def _build_adjacency_matrices(self, path: str):
        n_m = len(self.microbes_table)
        n_d = len(self.drugs_table)
        n_dis = len(self.disease_table)

        self.n_m = n_m
        self.n_d = n_d
        self.n_dis = n_dis

        print(f"[INFO] n_m={n_m}, n_d={n_d}, n_dis={n_dis}")

        self.A_mdis = np.zeros((n_m, n_dis), dtype=int)
        self.A_ddis = np.zeros((n_d, n_dis), dtype=int)

        # 微生物-疾病
        with open(self.microbe_dis_file, "r", encoding="utf-8") as f:
            print("[INFO] read:", self.microbe_dis_file)
            for line in tqdm(f, desc="microbe-disease"):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                microbe_id, disease_id = parts[0], parts[1]
                m_key = _safe_int_or_str(microbe_id)
                dis_key = _safe_int_or_str(disease_id)
                m_idx = self.microbes_table.get(m_key)
                dis_idx = self.disease_table.get(dis_key)
                if (m_idx is not None) and (dis_idx is not None):
                    self.A_mdis[m_idx - 1][dis_idx - 1] = 1

        # 药物-疾病
        with open(self.drug_dis_file, "r", encoding="utf-8") as f:
            print("[INFO] read:", self.drug_dis_file)
            for line in tqdm(f, desc="drug-disease"):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                drug_id, disease_id = parts[0], parts[1]
                d_key = _safe_int_or_str(drug_id)
                dis_key = _safe_int_or_str(disease_id)
                d_idx = self.drugs_table.get(d_key)
                dis_idx = self.disease_table.get(dis_key)
                if (d_idx is not None) and (dis_idx is not None):
                    self.A_ddis[d_idx - 1][dis_idx - 1] = 1

        # A_dmic 维度对齐到 (n_m, n_d)
        if self.A_dmic.shape != (n_m, n_d):
            if self.A_dmic.shape == (n_d, n_m):
                self.A_dmic = self.A_dmic.T
                print("[WARN] A_dmic transpose -> (n_m, n_d)")
            else:
                raise ValueError(f"A_dmic 维度异常：{self.A_dmic.shape}，期望 {(n_m, n_d)} 或 {(n_d, n_m)}")

        return self.A_mdis, self.A_ddis, self.A_dmic

    def _build_AH2_and_A2(self, A_mdis, A_ddis):
        n_m, n_dis = A_mdis.shape
        n_d = A_ddis.shape[0]

        # A_2 = A_mdis * A_ddis^T
        self.A_2 = np.dot(A_mdis, A_ddis.T)

        # A_H2：节点排列为 [m, dis, d]
        self.A_H2 = np.block([
            [np.zeros((n_m, n_m)), A_mdis, np.zeros((n_m, n_d))],
            [A_mdis.T, np.zeros((n_dis, n_dis)), A_ddis.T],
            [np.zeros((n_d, n_m)), A_ddis, np.zeros((n_d, n_d))],
        ])
        return self.A_H2, self.A_2

    # H1：节点排列 [m, d]
    def _build_A_H1(self, S_m, A_md, S_d):
        n_m, n_d = A_md.shape
        return np.block([
            [S_m, A_md],
            [A_md.T, S_d],
        ])

    def _merge_H1_H2(self, A_H1, A_H2, lambda_factor=0.3):
        """将 H1 嵌入到大图（m, dis, d）中并与 H2 融合。"""
        n_m, n_d = self.A_dmic.shape
        n_dis = self.n_dis
        n = A_H2.shape[0]

        H1_full = np.zeros((n, n), dtype=float)

        # S_m -> [0:n_m, 0:n_m]
        H1_full[0:n_m, 0:n_m] = A_H1[0:n_m, 0:n_m]

        # A_m_d -> [0:n_m, n_m+n_dis : n_m+n_dis+n_d]
        H1_full[0:n_m, n_m + n_dis: n_m + n_dis + n_d] = A_H1[0:n_m, n_m: n_m + n_d]

        # A_m_d^T -> [n_m+n_dis : , 0:n_m]
        H1_full[n_m + n_dis: n_m + n_dis + n_d, 0:n_m] = A_H1[n_m: n_m + n_d, 0:n_m]

        # S_d -> [n_m+n_dis : , n_m+n_dis : ]
        H1_full[n_m + n_dis: n_m + n_dis + n_d, n_m + n_dis: n_m + n_dis + n_d] = \
            A_H1[n_m: n_m + n_d, n_m: n_m + n_d]

        A = (1.0 - lambda_factor) * A_H2 + lambda_factor * H1_full
        return A


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load MDAD dataset and build adjacency matrices")
    parser.add_argument("--data-root", type=str, default="MDAD", help="Dataset root directory (default: ./MDAD)")
    parser.add_argument("--middle-dir", type=str, default=None, help="Directory to save middle .mat files")
    parser.add_argument("--no-middle-save", action="store_true", help="Disable saving middle .mat files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(script_dir, data_root))

    mode_data = mode_data_load(
        data_root,
        middle_dir=args.middle_dir,
        save_middle=(not args.no_middle_save),
    )
    print("[OK] Loaded.")
