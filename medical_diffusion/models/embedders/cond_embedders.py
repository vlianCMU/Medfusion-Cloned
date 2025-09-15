
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



