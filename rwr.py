'''
Author: EYEN-rick hostmrlan@outlook.com
Date: 2025-08-01 16:56:33
LastEditors: EYEN-rick hostmrlan@outlook.com
LastEditTime: 2025-08-01 16:57:35
FilePath: \BAN_RF\net_code\mode_ban_rf\rwr.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from tqdm import tqdm
def random_walk(T, start_idx, alpha=0.5, max_iter=100, tol=1e-6):
    '''
    随机游走，要求T为列归一
    '''
    n = T.shape[0]
    r = np.zeros(n)
    r[start_idx] = 1.0  # 起始节点
    prev_r = np.zeros(n)
    for _ in range(max_iter):
        r = (1 - alpha) * T @ r + alpha * np.eye(n)[start_idx]
        if np.linalg.norm(r - prev_r, 1) < tol:
            break
        prev_r = r.copy()
    return r  # 返回稳态向量
def rwr_matrix(T, seed_idx, alpha=0.5, max_iter=200, tol=1e-8, dtype=np.float64):
    """
    基于行随机 T 的带重启随机游走（矢量化），一次性对多个种子求稳态分布
    参数：
      - T: [N,N] 行归一的转移矩阵（row-stochastic）
      - seed_idx: 种子节点索引的一维数组（长度=K）
      - alpha: 重启概率（越大越“粘”种子）
      - max_iter, tol: 迭代上限与收敛阈值（L1范数）
    返回：
      - R: [K, N]，第 k 行是第 k 个种子在 N 个节点上的稳态分布
    """
    T = np.asarray(T, dtype=dtype)
    N = T.shape[0]
    K = len(seed_idx)

    # 种子矩阵 E：N×K，按列是 one-hot
    E = np.zeros((N, K), dtype=dtype)
    E[seed_idx, np.arange(K)] = 1.0

    # 初始化：可以直接从种子开始
    R = E.copy()
    TT = T.T  # 因为 T 是行归一，列向量更新要用 T^T

    # for _ in tqdm(range(max_iter)):
    for _ in range(max_iter):
        R_new = (1.0 - alpha) * (TT @ R) + alpha * E
        # 列归一（数值保险，理论上列和应为 1）
        col_sum = R_new.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0.0] = 1.0
        R_new = R_new / col_sum

        # 收敛判定（所有列最大 L1 差异）
        if np.max(np.sum(np.abs(R_new - R), axis=0)) < tol:
            R = R_new
            break
        R = R_new

    return R.T  # [K, N]