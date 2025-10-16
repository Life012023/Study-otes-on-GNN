import torch  
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用的函数式API
from torch_geometric.data import Data, HeteroData  # 导入PyG的图数据结构
from torch_geometric.nn import HGTConv, Linear, HeteroConv, GATConv  # 导入PyG的异构图神经网络层
import numpy as np  # 导入NumPy库
from sklearn.metrics import roc_auc_score, average_precision_score  # 导入评价指标
import pandas as pd  # 导入Pandas用于数据处理

# 定义异构图神经网络模型
class HeteroGNNDetoxifier(nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, compound_dim, protein_dim, herb_dim, metadata):
        super().__init__()  # 初始化父类

        self.hidden_channels = hidden_channels  # 隐藏层维度
        self.num_heads = num_heads  # 多头注意力头数
        
        # 特征投影层，将不同类型节点特征映射到统一维度
        self.compound_lin = Linear(compound_dim, hidden_channels)
        self.protein_lin = Linear(protein_dim, hidden_channels)
        self.herb_lin = Linear(herb_dim, hidden_channels)
        
        # HGT卷积层（异构图Transformer卷积）
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, num_heads, metadata)
            self.convs.append(conv)


        # 解毒概率预测头（全连接层+激活+Dropout+Sigmoid）
        self.detox_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_dict, edge_index_dict):
        # 特征投影
        x_dict['compound'] = self.compound_lin(x_dict['compound'])
        x_dict['protein'] = self.protein_lin(x_dict['protein']) 
        x_dict['herb'] = self.herb_lin(x_dict['herb'])
        
        # HGT卷积
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        
        return x_dict  # 返回每种节点类型的嵌入
    
    def predict_detox_probability(self, x_dict, herb_idx, detox_compound_indices, detox_target_indices, toxic_target_indices):
        """
        计算解毒概率: f(减毒成分嵌入, 减毒靶点嵌入, 毒性靶点嵌入)
        """
        # 获取药材嵌入
        herb_embedding = x_dict['herb'][herb_idx]
        
        # 获取该药材的减毒化合物嵌入（平均池化）
        if len(detox_compound_indices) > 0:
            detox_compound_embeddings = x_dict['compound'][detox_compound_indices]
            detox_compound_embedding = torch.mean(detox_compound_embeddings, dim=0)
        else:
            detox_compound_embedding = torch.zeros_like(herb_embedding)
        
        # 获取减毒靶点嵌入（平均池化）
        if len(detox_target_indices) > 0:
            detox_target_embeddings = x_dict['protein'][detox_target_indices]
            detox_target_embedding = torch.mean(detox_target_embeddings, dim=0)
        else:
            detox_target_embedding = torch.zeros_like(herb_embedding)
        
        # 获取毒性靶点嵌入（平均池化）
        if len(toxic_target_indices) > 0:
            toxic_target_embeddings = x_dict['protein'][toxic_target_indices]
            toxic_target_embedding = torch.mean(toxic_target_embeddings, dim=0)
        else:
            toxic_target_embedding = torch.zeros_like(herb_embedding)
        
        # 拼接特征并预测解毒概率
        combined_features = torch.cat([
            detox_compound_embedding,
            detox_target_embedding, 
            toxic_target_embedding
        ], dim=0)
        
        detox_prob = self.detox_predictor(combined_features.unsqueeze(0))
        return detox_prob.squeeze()  # 返回解毒概率

# 定义PU学习损失函数
class PULearningLoss(nn.Module):
    def __init__(self, prior=0.1):
        super().__init__()
        self.prior = prior  # 正样本先验概率
        
    def forward(self, positive_probs, unlabeled_probs):
        # 非负PU学习损失
        positive_loss = -torch.log(positive_probs + 1e-8).mean()
        unlabeled_loss = -torch.log(1 - unlabeled_probs + 1e-8).mean()
        
        # 非负风险估计器
        pu_loss = positive_loss * self.prior + torch.clamp(unlabeled_loss - self.prior * positive_loss, min=0)
        return pu_loss

# 构建异构图
def build_heterogeneous_graph(data):
    """
    构建异构图
    """
    hetero_data = HeteroData()
    
    # 节点特征
    hetero_data['compound'].x = torch.tensor(data['compound_features'], dtype=torch.float)
    hetero_data['protein'].x = torch.tensor(data['protein_features'], dtype=torch.float) 
    hetero_data['herb'].x = torch.tensor(data['herb_features'], dtype=torch.float)
    
    # 边索引
    hetero_data['herb', 'contains', 'compound'].edge_index = torch.tensor(
        data['herb_compound_edges'], dtype=torch.long
    )
    hetero_data['compound', 'targets', 'protein'].edge_index = torch.tensor(
        data['compound_protein_edges'], dtype=torch.long
    )
    hetero_data['protein', 'interacts', 'protein'].edge_index = torch.tensor(
        data['protein_protein_edges'], dtype=torch.long
    )
    
    return hetero_data

# 数据准备函数（需根据实际数据填充）
def prepare_data():
    """
    构造一个最小可运行的测试样本数据
    """
    # 假设有2个药材、3个化合物、2个蛋白
    compound_features = [
        [1.0, 0.0, 0.5],   # 化合物0
        [0.2, 1.0, 0.1],   # 化合物1
        [0.3, 0.3, 0.9],   # 化合物2
    ]
    protein_features = [
        [0.1, 0.2, 0.3],   # 蛋白0
        [0.4, 0.5, 0.6],   # 蛋白1
    ]
    herb_features = [
        [0.5, 0.5, 0.5],   # 药材0
        [0.1, 0.2, 0.3],   # 药材1
    ]

    # 边关系（注意：edge_index 形状为 [2, num_edges]，每列是 [src, dst]）
    herb_compound_edges = [
        [0, 0, 1],  # 药材0->化合物0, 药材0->化合物1, 药材1->化合物2
        [0, 1, 2]
    ]
    compound_protein_edges = [
        [0, 1, 2],  # 化合物0->蛋白0, 化合物1->蛋白1, 化合物2->蛋白0
        [0, 1, 0]
    ]
    protein_protein_edges = [
        [0, 1],     # 蛋白0<->蛋白1
        [1, 0]
    ]

    # 标注信息
    positive_herbs = [0]      # 药材0为正样本
    unlabeled_herbs = [1]     # 药材1为未标注
    negative_herbs = []       # 没有负样本

    # 药材对应的成分和靶点（这里简单写死）
    herb_detox_compounds = {
        0: [0, 1],   # 药材0含化合物0和1
        1: [2],      # 药材1含化合物2
    }
    herb_detox_targets = {
        0: [0],      # 药材0作用于蛋白0
        1: [1],      # 药材1作用于蛋白1
    }
    herb_toxic_targets = {
        0: [1],      # 药材0毒性靶点为蛋白1
        1: [0],      # 药材1毒性靶点为蛋白0
    }

    return {
        'compound_features': compound_features,
        'protein_features': protein_features,
        'herb_features': herb_features,
        'herb_compound_edges': herb_compound_edges,
        'compound_protein_edges': compound_protein_edges,
        'protein_protein_edges': protein_protein_edges,
        'positive_herbs': positive_herbs,
        'unlabeled_herbs': unlabeled_herbs,
        'negative_herbs': negative_herbs,
        'herb_detox_compounds': herb_detox_compounds,
        'herb_detox_targets': herb_detox_targets,
        'herb_toxic_targets': herb_toxic_targets
    }

# 训练模型主流程
def train_model():
    """
    训练模型
    """
    # 准备数据
    data_dict = prepare_data()
    hetero_data = build_heterogeneous_graph(data_dict)
    
    # 模型参数
    hidden_channels = 256
    num_heads = 8
    num_layers = 3
    compound_dim = len(data_dict['compound_features'][0])
    protein_dim = len(data_dict['protein_features'][0])
    herb_dim = len(data_dict['herb_features'][0])
    metadata = hetero_data.metadata()

    model = HeteroGNNDetoxifier(hidden_channels, num_heads, num_layers, 
                               compound_dim, protein_dim, herb_dim, metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    pu_loss_fn = PULearningLoss(prior=0.1)  # 假设正样本先验为10%
    
    # 训练循环
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # 前向传播
        x_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        
        # 计算正样本概率
        positive_probs = []
        for herb_idx in data_dict['positive_herbs']:
            detox_prob = model.predict_detox_probability(
                x_dict, herb_idx,
                data_dict['herb_detox_compounds'][herb_idx],
                data_dict['herb_detox_targets'][herb_idx], 
                data_dict['herb_toxic_targets'][herb_idx]
            )
            positive_probs.append(detox_prob)
        
        positive_probs = torch.stack(positive_probs)
        
        # 计算未标注样本概率
        unlabeled_probs = []
        for herb_idx in data_dict['unlabeled_herbs'] + data_dict['negative_herbs']:
            detox_prob = model.predict_detox_probability(
                x_dict, herb_idx,
                data_dict['herb_detox_compounds'][herb_idx],
                data_dict['herb_detox_targets'][herb_idx],
                data_dict['herb_toxic_targets'][herb_idx]
            )
            unlabeled_probs.append(detox_prob)
        
        unlabeled_probs = torch.stack(unlabeled_probs)
        
        # PU学习损失
        loss = pu_loss_fn(positive_probs, unlabeled_probs)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return model, hetero_data

# 评估和排序函数
def evaluate_and_rank(model, hetero_data, data_dict):
    """
    评估模型并对候选药材排序
    """
    model.eval()
    with torch.no_grad():
        x_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        
        # 为所有候选药材计算解毒概率
        detox_scores = []
        for herb_idx in data_dict['unlabeled_herbs']:
            detox_prob = model.predict_detox_probability(
                x_dict, herb_idx,
                data_dict['herb_detox_compounds'][herb_idx],
                data_dict['herb_detox_targets'][herb_idx],
                data_dict['herb_toxic_targets'][herb_idx]
            )
            detox_scores.append((herb_idx, detox_prob.item()))
        
        # 按解毒概率排序
        detox_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("候选药材解毒概率排序:")
        for rank, (herb_idx, score) in enumerate(detox_scores[:10], 1):
            print(f"排名 {rank}: 药材{herb_idx}, 解毒概率: {score:.4f}")
        
        return detox_scores

# 主执行流程
if __name__ == "__main__":
    # 训练模型
    print("开始训练模型...")
    model, hetero_data = train_model()
    print(model)  # 打印模型结构

    
    # 加载数据字典
    data_dict = prepare_data()
    
    # 评估和排序
    print("\n生成解毒概率排序...")
    ranked_herbs = evaluate_and_rank(model, hetero_data, data_dict)
    
    # 保存结果
    results_df = pd.DataFrame(ranked_herbs, columns=['herb_index', 'detox_probability'])
    print(results_df)
    # results_df.to_csv('detoxification_ranking.csv', index=False)
    # print("结果已保存到 detoxification_ranking.csv")