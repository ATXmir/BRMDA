import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import math

from scipy.io import savemat



'''
探讨微生物和药物关联

'''
def GIP_Calculate(M):     #计算微生物高斯核相似性
    l=np.size(M,axis=1)
    sm=[]
    m=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[:,i]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[:,i]-M[:,j]))**2))
    return m
def GIP_Calculate1(M):     #计算药物高斯核相似性
    l=np.size(M,axis=0)
    sm=[]
    m=np.zeros((l,l))
    km=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[i,:]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[i,:]-M[j,:]))**2))
    for i in range(l):
        for j in range(l):
            km[i,j]=1/(1+np.exp(-15*m[i,j]+math.log(9999)))
    return km

def gip_similarity(M, entity_axis=0, eps=1e-12):
    """
    计算 GIP 高斯相似度（不做额外挤压）。
    entity_axis: 实体维度（0=按行算；1=按列算）
    返回: [n_entities, n_entities] 相似矩阵，数值 ∈ (0, 1]
    """
    # 取出实体特征矩阵 X：每行是一个实体的交互轮廓
    X = M if entity_axis == 0 else M.T            # [n, d]
    X = np.asarray(X, dtype=np.float64)

    # 预计算 ||x_i||^2 与 Gram 矩阵
    sq = np.sum(X * X, axis=1)                    # [n]
    gram = X @ X.T                                # [n, n]
    # 成对平方距离：||xi-xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
    D2 = sq[:, None] + sq[None, :] - 2.0 * gram
    D2 = np.maximum(D2, 0.0)                      # 数值裁剪，避免负零

    # gamma = n / sum(||xi||^2)，加 eps 防除零
    gamma = X.shape[0] / (np.sum(sq) + eps)

    K = np.exp(-gamma * D2)                       # (0, 1]
    # 强化对称性与数值范围
    K = (K + K.T) * 0.5
    K = np.clip(K, 0.0, 1.0)
    return K

def gip_similarity_with_logistic(M, entity_axis=0, k=15.0, target=9999.0, eps=1e-12):
    """
    在 GIP 基础上增加可选的 logistic 挤压，保持两侧一致性时才使用。
    σ(-k * K + ln(target)) 将相似度挤压到 ~[1/(1+target), ~1)
    """
    K = gip_similarity(M, entity_axis=entity_axis, eps=eps).astype(np.float64)
    b = np.log(target)
    Z = -k * K + b
    KM = 1.0 / (1.0 + np.exp(Z))
    KM = (KM + KM.T) * 0.5
    KM = np.clip(KM, 0.0, 1.0)
    return KM

class mode_data_load:
    def __init__(self, path):
        print("in path",path)
        '''
            训练批次主要的划分方式在于使用已知关联,我们将已知关联作为数据集时会同步修改
            已知关联，并将其对应的邻接矩阵进行清零
            使用MDAD
        '''

        self.load_file_data(path)
        A_mdis, A_ddis ,A_dmic= self.build_adjacency_matrices(path)
        # print("微生物-疾病关联矩阵形状：", A_mdis.shape)
        # print("药物-疾病邻关联阵形状：", A_ddis.shape)
        # print("药物-微生物关联矩阵形状：", A_dmic.shape)

        self.save_data_middle_path = os.path.abspath(os.path.join(path, '..'))
        self.save_data_middle_path = os.path.join(self.save_data_middle_path,'middle')
        savemat( os.path.join(self.save_data_middle_path,'A_mdis.mat'), {'A_mdis': np.array(A_mdis)})#保存生成的临界矩阵到临时文件
        savemat( os.path.join(self.save_data_middle_path,'A_ddis.mat'), {'A_ddis': np.array(A_ddis)})#保存生成的临界矩阵到临时文件
        savemat( os.path.join(self.save_data_middle_path,'A_dmic.mat'), {'A_dmic': np.array(A_dmic)})#保存生成的临界矩阵到临时文件




        self.build_AH2_and_A2(A_mdis, A_ddis)
        #构建潜在样本
        # print("桥接矩阵AH2形状：", self.A_H2.shape)
        # print("桥接矩阵A2形状：", self.A_2.shape)
        savemat( os.path.join(self.save_data_middle_path,'example.mat'), {'example_matrix': np.array(self.A_2)})#保存生成的临界矩阵到临时文件
        #此处进行数据集嵌入
        # —— 计算相似度（建议两侧都用同一套）
        S_m = gip_similarity(self.A_dmic, entity_axis=0)  # 微生物相似（按行）
        S_d = gip_similarity(self.A_dmic, entity_axis=1)  # 药物相似（按列）
        # —— 构 H1 并与 H2 融合（见 D 补丁）
        A_H1 = self.build_A_H1(S_m, self.A_dmic, S_d)
        # print("H1 的矩阵大小为 ",A_H1.shape)
        ##输入关联矩阵和相似性
        A = self.merge_H1_H2(A_H1, self.A_H2, lambda_factor=0.1)
        A = np.maximum(A, 0.0)

        # 2) 给零度节点自环（或全图加极小对角）
        deg = A.sum(axis=1, keepdims=True)
        zero = (deg == 0)
        A[zero.squeeze(), zero.squeeze()] = 1.0

        # 3) 行归一
        deg = A.sum(axis=1, keepdims=True)
        self.T = (A / deg).astype(np.float32)
        pass
    ###
    def data_excel_read(self,read_path,label_index_name):
        '''
            返回建立索引
        '''
        # print("path : ",read_path, "read name: ",label_index_name)
        # 检查文件是否存在
        if not os.path.exists(read_path):
            print(f"错误：文件不存在 - {read_path}")
            return {}
        try:
            # 先尝试openpyxl引擎
            df = pd.read_excel(read_path, sheet_name=0, engine="openpyxl")
        except Exception as e:
            print(f"使用openpyxl引擎失败: {e}")
            try:
                # 尝试其他引擎
                df = pd.read_excel(read_path, sheet_name=0, engine=None)
            except Exception as e2:
                print(f"所有引擎都失败: {e2}")
                return {}
        # 检查指定的列是否存在
        if label_index_name not in df.columns:
            print(f"错误：列 '{label_index_name}' 不存在于文件中")
            print(f"可用列: {list(df.columns)}")
            return {}
        # df = pd.read_excel(read_path, sheet_name=0, engine="openpyxl")  # 第一个表
        index_table = {idx+1:id_ for idx, id_ in enumerate(tqdm(df[label_index_name]))}
        return index_table
    
    # def _t

    def data_type_segmentation(self):
        """
            切分数据集,
        """
        self.train_base
        self.test_base
        pass
    ##  加载文件数据
    def load_file_data(self ,path):
        self.microbes_table = self.data_excel_read(os.path.join(path, "microbes.xlsx"),"Microbe_ID")
        self.disease_table = self.data_excel_read(os.path.join(path, "disease.xlsx"),"Unnamed: 0")
        self.drugs_table = self.data_excel_read(os.path.join(path, "drugs.xlsx"),"Drug_ID")
        # print(self.microbes_table)
        self.drug_dis_file = os.path.join(path, "drug_with_disease.txt")
        self.microbe_dis_file = os.path.join(path, "microbe_with_disease.txt")
        self.A_dmic = np.loadtxt(os.path.join(path, "drug_microbe_matrix.txt"))  #单独加载药物和微生物邻接矩阵
        ##获取全部已知关联
        # self.all_know = np.loadtxt(os.path.join(path, "disease_microbe_known.txt"))  # 已知关联索引（序号从1开始）疾病和微生物

    #数据分类_5折
    def fold_5(self):
        pass
    



        
    def build_adjacency_matrices(self,path):
        """
        从提供的 CSV 和 txt 文件中构建微生物-疾病和药物-疾病邻接矩阵 

        参数:
            microbes_csv: 微生物 ID 文件路径 (.csv)
            diseases_csv: 疾病 ID 文件路径 (.csv)
            drugs_csv: 药物 ID 文件路径 (.csv)
            microbe_dis_file: 微生物与疾病关联 (.txt)
            drug_dis_file: 药物与疾病关联 (.txt)
        返回:
            A_mdis: 微生物-疾病邻接矩阵 [n_m, n_dis]
            A_ddis: 药物-疾病邻接矩阵 [n_d, n_dis]
        """


        #加载已知矩阵


        n_m = len(self.microbes_table)
        n_d = len(self.drugs_table )
        n_dis = len(self.disease_table)

        print(n_m,n_d,n_dis)
        self.n_m = n_m 
        self.n_d = n_d 
        self.n_dis = n_dis
        self.A_mdis = np.zeros((n_m, n_dis), dtype=int)
        self.A_ddis = np.zeros((n_d, n_dis), dtype=int)

        # 填充微生物-疾病
        with open(self.microbe_dis_file, "r") as f:
            print("read ok , path : ",self.microbe_dis_file)
            for line in tqdm(f):
                microbe_id, disease_id = line.strip().split()
                m_idx = self.microbes_table.get(int(microbe_id))
                d_idx = self.disease_table.get(int(disease_id))
                if m_idx is not None and d_idx is not None:
                    self.A_mdis[m_idx-1][d_idx-1] = 1
        # 填充药物-疾病
        with open(self.drug_dis_file, "r") as f:
            print("read ok , path : ",self.drug_dis_file)
            for line in tqdm(f):
                drug_id, disease_id = line.strip().split()
                d_idx = self.drugs_table.get(int(drug_id))
                dis_idx = self.disease_table.get(int(disease_id))
                if d_idx is not None and dis_idx is not None:
                    self.A_ddis[d_idx-1][dis_idx-1] = 1
        if self.A_dmic.shape != (n_m, n_d):
            if self.A_dmic.shape == (n_d, n_m):
                self.A_dmic = self.A_dmic.T
                print("数据倒置 药物微生物改为微生物，药物")
            else:
                raise ValueError(f"A_dmic 维度异常：当前 {self.A_dmic.shape}，期望 {(n_m, n_d)} 或 {(n_d, n_m)}")
        return self.A_mdis, self.A_ddis,self.A_dmic
           ###
    def build_AH2_and_A2(self, A_mdis, A_ddis):
        """
        构建异构图邻接矩阵 A_H2 和间接桥接矩阵 A2

        参数:
        A_mdis: np.ndarray, 微生物-疾病邻接矩阵 [n_m, n_dis]
        A_ddis: np.ndarray, 药物-疾病邻接矩阵 [n_d, n_dis]

        返回:
        A_H2: 异构邻接矩阵，包含微生物、疾病、药物三类节点的结构连接
        A_2: 微生物-药物之间的疾病桥接打分矩阵
        """
        n_m, n_dis = A_mdis.shape
        n_d = A_ddis.shape[0]

        # A2 = M * D^T
        self.A_2 = np.dot(A_mdis, A_ddis.T)

        # 构建异构邻接矩阵 A_H2
        self.A_H2 = np.block([
        [np.zeros((n_m, n_m)), A_mdis, np.zeros((n_m, n_d))],
        [A_mdis.T, np.zeros((n_dis, n_dis)), A_ddis.T],
        [np.zeros((n_d, n_m)), A_ddis, np.zeros((n_d, n_d))]
        ])
        return self.A_H2,self.A_2

    def build_A_H1(self, S_m, A1, S_d):
        """
        S_m 左上角矩阵，微生物
        A1  关联矩阵
        S_d 右下角矩阵，药物
        """
        n_m, n_d = A1.shape
        A_H1 = np.block([
            [S_m, A1],
            [A1.T, S_d]
        ])
        return A_H1
    # ========== 7. 构建转移矩阵 ==========
    def build_transition_matrix(self,A):
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        # T = A / row_sum
        T = A / (A.sum(axis=1, keepdims=True) + 1e-6)
        return T
    #矩阵融合
    def merge_H1_H2(self,A_H1, A_H2, lambda_factor=0.3):
        """
        A_H1为异构网络矩阵
        A_H2为三类节点之间关联关系的完整临界矩阵
        """
        """
        A_H1: [[S_m, A_m_d],
               [A_m_d^T, S_d]]  —— 节点排列为 [m, d]
        A_H2: 三类节点大图，排列为 [m, dis, d]
        将 A_H1 对齐嵌入到 A_H2 的 [m] 与 [d] 两个子块，并在 m↔d 跨越疾病块放置边
        """
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
        # print("A_H1_full:", H1_full.shape)  # 预计 (n_m+n_dis+n_d, n_m+n_dis+n_d)
        A = (1.0 - lambda_factor) * A_H2 + lambda_factor * H1_full
        # print("A_final:", A.shape)  # 预计 (1655,1655)

        # n = A_H2.shape[0]
        # A_H1_up = np.zeros((n, n))
        # h1_shape = A_H1.shape[0]
        # A_H1_up[:h1_shape, :h1_shape] = A_H1
        # A = A_H2 + lambda_factor * A_H1_up
        return A



        pass
if __name__ == "__main__":
    mode_data = mode_data_load("..\..\data\source")