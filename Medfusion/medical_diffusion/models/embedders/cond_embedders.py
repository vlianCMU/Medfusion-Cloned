
import torch.nn as nn
import torch 
from monai.networks.layers.utils import get_act_layer

class LabelEmbedder(nn.Module):
    def __init__(self, emb_dim=32, num_classes=2, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_classes, emb_dim)

        # self.embedding = nn.Embedding(num_classes, emb_dim//4)
        # self.emb_net = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     get_act_layer(act_name),
        #     nn.Linear(emb_dim, emb_dim)
        # )

    def forward(self, condition):
        c = self.embedding(condition) #[B,] -> [B, C]
        # c = self.emb_net(c)
        # c = self.emb_net(condition[:,None].float())
        # c = (2*condition-1)[:, None].expand(-1, self.emb_dim).type(torch.float32)
        return c
    
class FundusDiseaseEmbedder(nn.Module):
    """眼底疾病条件嵌入器"""
    def __init__(self, emb_dim=512, num_diseases=6, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_diseases = num_diseases
        
        # 疾病条件嵌入网络
        self.disease_embedding = nn.Sequential(
            nn.Linear(num_diseases, emb_dim),
            get_act_layer(act_name),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, condition):
        # condition: [B, num_diseases] 疾病multi-hot向量
        disease_emb = self.disease_embedding(condition)  # [B, emb_dim]
        return disease_emb

class FundusConditionEmbedder(nn.Module):
    """
    综合条件嵌入器：
    - 疾病: multi-hot 向量
    - 眼别: 单值 (0=右,1=左)
    """
    def __init__(self, emb_dim=512, num_diseases=10, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_diseases = num_diseases

        # 疾病条件嵌入网络（multi-hot）
        self.disease_embedding = nn.Sequential(
            nn.Linear(num_diseases, emb_dim),
            get_act_layer(act_name),
            # nn.Dropout(0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        # 眼别与质量：独立Embedding
        self.eye_embedding = nn.Embedding(2, emb_dim)

        # 组合映射：融合两部分
        self.combine = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            get_act_layer(act_name),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, condition):
        """
        condition: tuple of (disease_vec, eye_side)
        disease_vec: [B, num_diseases]  multi-hot 疾病
        eye_side: [B] 0/1
        """
        disease_vec, eye_side = condition
        
        disease_emb = self.disease_embedding(disease_vec)
        eye_emb = self.eye_embedding(eye_side)
        
        # 拼接后融合
        cond_emb = torch.cat([disease_emb, eye_emb], dim=-1)
        cond_emb = self.combine(cond_emb)
        return cond_emb
    
class SynFundusConditionEmbedder(nn.Module):
    """
    条件嵌入器：
    - 输入: [B, num_labels]（15维标签向量）
    - 输出: [B, emb_dim]
    """
    def __init__(self, emb_dim=512, num_labels=15, act_name=("SWISH", {})):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_labels = num_labels

        self.embedding = nn.Sequential(
            nn.Linear(num_labels, emb_dim),
            get_act_layer(act_name),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, condition):
        """
        condition: tensor [B, num_labels]，直接来自 batch['labels']
        """
        return self.embedding(condition)




