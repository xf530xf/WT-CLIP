import torch
from torch import nn
from torch.nn import functional as F




class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge



class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out
    



# class MLFusion(nn.Module):
#     def __init__(self, norm = nn.BatchNorm2d, act=nn.ReLU):
#         super().__init__()
#         self.fusi_conv = nn.Sequential(
#             nn.Conv2d(1024, 256, 1,bias = False),
#             norm(256),
#             act(),
#         )

#         self.attn_conv = nn.ModuleList()
#         for i in range(4):
#             self.attn_conv.append(nn.Sequential(
#                 nn.Conv2d(256, 256, 1,bias = False),
#                 norm(256),
#                 act(),
#             ))

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, feature_list):
#         fusi_feature = torch.cat(feature_list, dim = 1).contiguous()
#         fusi_feature = self.fusi_conv(fusi_feature)

#         for i in range(4):
#             x = feature_list[i]
#             attn = self.attn_conv[i](x)
#             attn = self.pool(attn)
#             attn = self.sigmoid(attn)

#             x = attn * x + x
#             feature_list[i] = x
        
#         return feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]
    
    

import torch
from torch import nn
import math

class MLFusion(nn.Module):
    def __init__(self, in_channels=768, mid_channels=256, out_channels=768, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            norm(mid_channels),
            act(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            norm(out_channels),
            act()
        )

    def forward(self, feature_list):
        # feature_list: List of 11 tensors, each of shape [1025, 4, 768]
        B = feature_list[0].shape[1]
        C = feature_list[0].shape[2]
        S = feature_list[0].shape[0]

        # Calculate H, W to reshape (pad if needed)
        H = int(math.ceil(S ** 0.5))  # e.g., 33
        W = H
        new_len = H * W  # e.g., 1089

        fused_list = []

        for f in feature_list:
            if f.shape[0] < new_len:
                pad_len = new_len - f.shape[0]
                pad = torch.zeros(pad_len, B, C, device=f.device, dtype=f.dtype)
                f = torch.cat([f, pad], dim=0)  # [new_len, B, C]

            # reshape to [H, W, B, C] → [B, C, H, W]
            f = f.view(H, W, B, C).permute(2, 3, 0, 1).contiguous()
            # apply fusion conv
            f=f.float()
            f_fused = self.fusion(f)  # [B, C, H, W]

            # reshape back: [B, C, H, W] → [H, W, B, C] → [new_len, B, C]
            f_fused = f_fused.permute(2, 3, 0, 1).contiguous().view(new_len, B, C)

            f_fused = f_fused[:S]  # remove padding
            fused_list.append(f_fused)

        # stack all: [11, 1025, 4, 768]
        return torch.stack(fused_list, dim=0)

# if __name__ == "__main__":
#     x = torch.randn(3, 520, 520)   # (C, H, W)
#     x = x.unsqueeze(0)             # 添加 batch 维度，变成 (1, 3, 520, 520)

#     model = MEEM(in_dim=3, hidden_dim=32)

#     y = model(x)                   # 输出是 (1, 3, 520, 520)
#     y = y.squeeze(0)              # 去掉 batch 维度 -> (3, 520, 520)

#     print(y.shape)                # 输出 torch.Size([3, 520, 520])



# if __name__ == "__main__":
#     feature_list = [torch.randn(1025, 4, 768) for _ in range(11)]
#     model = MLFusion()
#     fused = model(feature_list)
#     print(fused.shape)  # ➜ torch.Size([11, 1025, 4, 768])
    




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List


# class AdaptiveFeatureFusionModule(nn.Module):
#     """
#     多特征自适应融合模块
    
#     输入: 11个形状为(1025, 4, 768)的向量列表
#     输出: 形状为(11, 1025, 4, 768)的融合特征
#     """
    
#     def __init__(self, 
#                  num_features: int = 11, 
#                  feature_dim: int = 768,
#                  reduction_ratio: int = 16,
#                  fusion_type: str = 'channel_spatial'):
#         """
#         Args:
#             num_features: 特征向量数量，默认11
#             feature_dim: 特征维度，默认768
#             reduction_ratio: 注意力机制的降维比例
#             fusion_type: 融合类型 ('channel_spatial', 'global', 'hierarchical')
#         """
#         super(AdaptiveFeatureFusionModule, self).__init__()
        
#         self.num_features = num_features
#         self.feature_dim = feature_dim
#         self.fusion_type = fusion_type
        
#         # 全局平均池化用于特征统计
#         self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        
#         # 权重生成网络 - 基于全局特征统计
#         self.weight_generator = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // reduction_ratio, feature_dim // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // reduction_ratio, 1),
#             nn.Sigmoid()
#         )
        
#         # 通道注意力机制
#         self.channel_attention = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(feature_dim // reduction_ratio, feature_dim),
#             nn.Sigmoid()
#         )
        
#         # 空间注意力机制
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         # 特征增强模块
#         self.feature_enhancement = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1)
#         )
        
#         # 层归一化
#         self.layer_norm = nn.LayerNorm(feature_dim)
        
#         # 可学习的温度参数
#         self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
#     def compute_feature_importance(self, features: List[torch.Tensor]) -> torch.Tensor:
#         """
#         计算每个特征的重要性权重
        
#         Args:
#             features: 特征列表，每个元素形状为(1025, 4, 768)
            
#         Returns:
#             权重张量，形状为(11,)
#         """
#         importance_scores = []
        
#         for feature in features:
#             # feature shape: (1025, 4, 768)
#             # 重新排列维度以便池化操作: (4, 768, 1025) -> (4, 768, 32, 32) 近似
#             hw = feature.shape[0]  # 1025
#             batch_size = feature.shape[1]  # 4
#             channels = feature.shape[2]  # 768
            
#             # 将空间维度重新整理
#             h = w = int(hw ** 0.5)  # 约32
#             if h * w != hw:
#                 # 如果不是完美平方数，进行填充
#                 pad_size = h * w - hw
#                 feature_padded = F.pad(feature, (0, 0, 0, 0, 0, pad_size))
#                 feature_reshaped = feature_padded.view(h, w, batch_size, channels)
#             else:
#                 feature_reshaped = feature.view(h, w, batch_size, channels)
            
#             # 转换为 (batch_size, channels, h, w) 格式
#             feature_reshaped = feature_reshaped.permute(2, 3, 0, 1)
            
#             # 计算全局平均池化和最大池化
#             avg_pool = self.global_avg_pool(feature_reshaped.unsqueeze(-1)).squeeze(-1)  # (4, 768, 1, 1)
#             max_pool = self.global_max_pool(feature_reshaped.unsqueeze(-1)).squeeze(-1)  # (4, 768, 1, 1)
            
#             # 合并统计信息
#             combined_pool = torch.cat([avg_pool.squeeze(-1).squeeze(-1), 
#                                      max_pool.squeeze(-1).squeeze(-1)], dim=1)  # (4, 1536)
            
#             # 计算重要性得分
#             importance = self.weight_generator(combined_pool)  # (4, 1)
#             importance_scores.append(importance.mean(dim=0))  # 对batch维度求平均
        
#         # 堆叠所有重要性得分
#         importance_weights = torch.stack(importance_scores, dim=0)  # (11, 1)
        
#         # 使用softmax进行归一化，添加温度参数
#         importance_weights = F.softmax(importance_weights.squeeze(-1) / self.temperature, dim=0)
        
#         return importance_weights
    
#     def apply_attention(self, feature: torch.Tensor) -> torch.Tensor:
#         """
#         对单个特征应用注意力机制
        
#         Args:
#             feature: 输入特征，形状为(1025, 4, 768)
            
#         Returns:
#             增强后的特征，形状为(1025, 4, 768)
#         """
#         hw, batch_size, channels = feature.shape
        
#         # 重新排列维度进行注意力计算
#         feature_for_attention = feature.permute(1, 0, 2)  # (4, 1025, 768)
        
#         # 计算通道注意力
#         avg_pool = torch.mean(feature_for_attention, dim=1)  # (4, 768)
#         max_pool = torch.max(feature_for_attention, dim=1)[0]  # (4, 768)
        
#         channel_att_input = torch.cat([avg_pool, max_pool], dim=1)  # (4, 1536)
#         channel_weights = self.channel_attention(channel_att_input)  # (4, 768)
        
#         # 应用通道注意力
#         feature_enhanced = feature_for_attention * channel_weights.unsqueeze(1)  # (4, 1025, 768)
        
#         # 特征增强
#         feature_enhanced = self.feature_enhancement(feature_enhanced)
        
#         # 残差连接和层归一化
#         feature_enhanced = self.layer_norm(feature_enhanced + feature_for_attention)
        
#         # 恢复原始维度顺序
#         return feature_enhanced.permute(1, 0, 2)  # (1025, 4, 768)
    
#     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
#         """
#         前向传播
        
#         Args:
#             features: 输入特征列表，11个形状为(1025, 4, 768)的张量
            
#         Returns:
#             融合后的特征，形状为(11, 1025, 4, 768)
#         """
#         assert len(features) == self.num_features, f"Expected {self.num_features} features, got {len(features)}"
        
#         # 计算特征重要性权重
#         importance_weights = self.compute_feature_importance(features)
        
#         # 对每个特征应用注意力机制
#         enhanced_features = []
#         for i, feature in enumerate(features):
#             enhanced_feature = self.apply_attention(feature)
#             # 应用重要性权重
#             weighted_feature = enhanced_feature * importance_weights[i]
#             enhanced_features.append(weighted_feature)
        
#         # 堆叠所有特征
#         output = torch.stack(enhanced_features, dim=0)  # (11, 1025, 4, 768)
        
#         return output
    
#     def get_feature_weights(self, features: List[torch.Tensor]) -> torch.Tensor:
#         """
#         获取特征权重（用于可视化和分析）
        
#         Args:
#             features: 输入特征列表
            
#         Returns:
#             特征权重，形状为(11,)
#         """
#         return self.compute_feature_importance(features)


# # 使用示例
# if __name__ == "__main__":
#     # 创建模拟数据
#     features = [torch.randn(1025, 4, 768) for _ in range(11)]
    
#     # 创建融合模块
#     fusion_module = AdaptiveFeatureFusionModule(
#         num_features=11,
#         feature_dim=768,
#         reduction_ratio=16,
#         fusion_type='channel_spatial'
#     )
    
#     # 前向传播
#     output = fusion_module(features)
#     print(f"Input: {len(features)} features of shape {features[0].shape}")
#     print(f"Output shape: {output.shape}")
    
#     # 获取特征权重
#     weights = fusion_module.get_feature_weights(features)
#     print(f"Feature weights: {weights}")
    
#     # 计算参数数量
#     total_params = sum(p.numel() for p in fusion_module.parameters())
#     print(f"Total parameters: {total_params:,}")
    
#     # 测试梯度反向传播
#     loss = output.sum()
#     loss.backward()
#     print("Gradient computation successful!")
    



import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_dim=768, num_features=11, hidden_dim=128):
        super(AdaptiveFeatureFusion, self).__init__()
        self.num_features = num_features
        

        self.attention_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feature_list):
        """
        feature_list: List of 11 tensors, each of shape (1025, 4, 768), dtype float16
        Returns:
            Tensor of shape (11, 1025, 4, 768), same dtype
        """
        assert len(feature_list) == self.num_features, f"Expected {self.num_features} features"

        scores = []

        for feat in feature_list:
            # feat: (1025, 4, 768)
            B = feat.shape[1]
            feat_fp32 = feat.to(dtype=torch.float32)  # 转换为 float32 用于精度更好的 MLP 运算
            global_desc = feat_fp32.mean(dim=(0, 1))  # 对 hw 和 batch 平均 → (768,)
            score = self.attention_mlp(global_desc)   # → (1,)
            scores.append(score)

        # Stack 得到 shape: (11, 1)
        scores_tensor = torch.stack(scores, dim=0)  # (11, 1)
        weights = F.softmax(scores_tensor.squeeze(-1), dim=0)  # (11,)

        # 加权融合：为每个特征乘以对应权重
        fused_features = []
        for i, feat in enumerate(feature_list):
            weight = weights[i].to(feat.device).to(dtype=feat.dtype)  # 转回 float16
            fused = feat * weight.view(1, 1, 1)  # 广播乘
            fused_features.append(fused)

        # 输出 shape: (11, 1025, 4, 768)
        output = torch.stack(fused_features, dim=0)
        return output
    






import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class FeatureEdgeEnhancedCAM(nn.Module):
    def __init__(self, edge_weight=0.3):
        super().__init__()
        self.edge_weight = edge_weight

        # 定义 Sobel 核：用于边缘提取
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))  # (1,1,3,3)
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, features: torch.Tensor, cam: torch.Tensor):
        """
        features: [1025, 4, 768]
        cam: [1, 32, 32]
        """
        assert features.shape == (1025, 4, 768)
        assert cam.shape == (1, 32, 32)

        # 去掉 CLS token
        patch_tokens = features[1:, :, :]  # [1024, 4, 768]

        # 平均多个 token 通道（通常是取第0个，或平均）
        patch_tokens = patch_tokens.mean(dim=1)  # [1024, 768]

        # reshape 成空间结构
        fmap = patch_tokens.view(32, 32, 768).permute(2, 0, 1).unsqueeze(0)  # [1, 768, 32, 32]

        # 将特征图做降维为单通道（可用平均池）
        fmap_1c = fmap.mean(dim=1, keepdim=True)  # [1, 1, 32, 32]

        # 提取边缘信息（Sobel）
        edge_x = F.conv2d(fmap_1c, self.sobel_x, padding=1)
        edge_y = F.conv2d(fmap_1c, self.sobel_y, padding=1)
        edge_map = torch.sqrt(edge_x**2 + edge_y**2)  # [1, 1, 32, 32]

        # 归一化边缘图
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)

        # 融合边缘图与CAM
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        enhanced_cam = cam_norm * (1 - self.edge_weight) + edge_map * self.edge_weight

        # 最终归一化
        enhanced_cam = (enhanced_cam - enhanced_cam.min()) / (enhanced_cam.max() - enhanced_cam.min() + 1e-8)

        return enhanced_cam  # [1, 32, 32]
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeEnhancedCAM(nn.Module):
    def __init__(self):
        super(EdgeEnhancedCAM, self).__init__()
        # 简单的卷积用于提取边缘（可替换为 Sobel）
        self.edge_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        sobel_kernel = torch.tensor([[[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2,  1]]], dtype=torch.float32).unsqueeze(0)
        self.edge_conv.weight = nn.Parameter(sobel_kernel, requires_grad=False)

        # 映射到32x32尺寸
        self.proj = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 32)
        )

    def forward(self, features_fp16: torch.Tensor, cam_fp32: torch.Tensor):
        # 1. 转为 float32
        features = features_fp16.float()  # [1025, 4, 768]

        # 2. 平均池化空间维度
        pooled = features.mean(dim=1)  # [1025, 768]

        # 3. 映射为边缘图的初始值
        mapped = self.proj(pooled)  # [1025, 1024]
        mapped = mapped.mean(dim=0).view(1, 1, 32, 32)  # [1, 1, 32, 32]

        # 4. 提取边缘
        edge_map = self.edge_conv(mapped)  # [1, 1, 32, 32]

        # 5. 标准化
        edge_map = F.relu(edge_map)
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-6)

        # 6. 融合
        cam = cam_fp32.unsqueeze(1)  # [1, 1, 32, 32]
        enhanced = cam + 0.5 * edge_map  # 权重可调
        enhanced = enhanced.squeeze(1)  # [1, 32, 32]

        return enhanced.float()  # 输出保持 float32

if __name__ == "__main__":

    # x1=torch.randn(1025,4,768)
    # x2=torch.randn(1,32,32)

    # enhancer = FeatureEdgeEnhancedCAM(edge_weight=0.3)
    # enhanced_cam = enhancer(x1, x2)  # 输出 shape [1, 32, 32]
    # print(enhanced_cam.squeeze(0).shape)


    model = EdgeEnhancedCAM()

    # 输入张量
    x_feat = torch.randn(1025, 4, 768, dtype=torch.float16)
    cam = torch.randn(1, 32, 32, dtype=torch.float32)

    # 输出
    enhanced_cam = model(x_feat, cam)
    print(enhanced_cam.shape, enhanced_cam.dtype)