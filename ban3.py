
import torch
import torch.nn as nn
import torch.nn.functional as F



# === 以下省略图结构与T矩阵构建部分，请结合前文件使用 ===

# ========== BAN 模型（简化三模态双线性交互） ==========
class BilinearAttention(nn.Module):
    def __init__(self, dim):
        super(BilinearAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.bias = nn.Parameter(torch.Tensor(dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x1, x2):
        # 输入 [batch, dim], 输出 [batch, dim]
        return F.relu(torch.matmul(x1, self.weight) * x2 + self.bias)

class BANModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, p=0.1):
        super(BANModel, self).__init__()
        # self.proj_m = nn.Linear(in_dim, hidden_dim)
        # self.proj_d = nn.Linear(in_dim, hidden_dim)
        # self.proj_dis = nn.Linear(in_dim, hidden_dim)

        self.proj_m = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.proj_d = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.proj_dis = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

        self.att_md = BilinearAttention(hidden_dim)
        self.att_mdise = BilinearAttention(hidden_dim)
        self.att_ddise = BilinearAttention(hidden_dim)

        # self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        # self.out = nn.Linear(hidden_dim, out_dim)

        self.bn = nn.BatchNorm1d(hidden_dim * 3)
        self.fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, r_m, r_d, r_dis):
        # x_m = self.proj_m(r_m)
        # x_d = self.proj_d(r_d)
        # x_dis = self.proj_dis(r_dis)
        #
        # z1 = self.att_md(x_m, x_d)
        # z2 = self.att_mdise(x_m, x_dis)
        # z3 = self.att_ddise(x_d, x_dis)
        #
        # z = torch.cat([z1, z2, z3], dim=1)
        # z_mid = F.relu(self.fc(z))
        # out = self.out(z_mid)
        # return out, z_mid  # 返回输出和中间特征向量

        x_m, x_d, x_dis = self.proj_m(r_m), self.proj_d(r_d), self.proj_dis(r_dis)
        z = torch.cat([self.att_md(x_m, x_d),
                       self.att_mdise(x_m, x_dis),
                       self.att_ddise(x_d, x_dis)], dim=1)
        z = self.bn(z)
        z_mid = F.relu(self.fc(z))
        z_mid = self.drop(z_mid)
        return self.out(z_mid), z_mid


