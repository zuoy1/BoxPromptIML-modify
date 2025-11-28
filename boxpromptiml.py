import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from TinyViT.models.tiny_vit import tiny_vit_5m_224
import numpy as np

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    

class Fusion(nn.Module):
    def __init__(self,channel):
        super(Fusion, self).__init__()
        self.conv1_1 = Conv1x1(channel + channel, channel)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((lf, hf), dim=1)
        x = self.conv1_1(x)

        return x


class BoxPromptIML(nn.Module):
    def __init__(self):
        super(BoxPromptIML, self).__init__()
        # Using tiny_vit_5m_224 as the backbone
        self.backbone = tiny_vit_5m_224(pretrained=True)
        self.fusion = GateFusionModule()
        self.conv1x1_1 = nn.Conv2d(128, 64, 1, bias=False)
        self.conv1x1_2 = nn.Conv2d(160, 64, 1, bias=False)
        self.conv1x1_3 = nn.Conv2d(320, 64, 1, bias=False)
        self.conv1x1_4 = nn.Conv2d(320, 64, 1, bias=False)
        
        self.upconv = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # 14->28
                nn.BatchNorm2d(64),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # 7->14
                nn.BatchNorm2d(64),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # 7->14
                nn.BatchNorm2d(64),
                nn.GELU(),
            )
        ])

        self.ex_attn = ExternalAttention(64)

        self.last_conv1x1 = nn.Conv2d(64, 1, 1, bias=False)


    def forward(self, x):

        image_shape = x.size()[2:]

        features, resolutions = self.backbone.get_intermediate_features(x)
        feat_maps = []
        # Processing features at all levels
        for feat, (H, W) in zip(features, resolutions):
            # Convert to shape [B, C, H, W]
            batch_size = feat.shape[0]
            feat_2d = feat.view(batch_size, H, W, -1).permute(0, 3, 1, 2)
            feat_maps.append(feat_2d)

        """
            tiny_vit_5m_224:
            feat_2d shape:torch.Size([1, 64, 56, 56])
            feat_2d shape:torch.Size([1, 128, 28, 28])
            feat_2d shape:torch.Size([1, 160, 14, 14])
            feat_2d shape:torch.Size([1, 320, 7, 7])
            feat_2d shape:torch.Size([1, 320, 7, 7]) 
        """

        x0 = feat_maps[0]
        x1 = feat_maps[1]
        x2 = feat_maps[2]
        x3 = feat_maps[3]
        x4 = feat_maps[4]

        x1 = self.conv1x1_1(x1)
        x2 = self.conv1x1_2(x2)
        x3 = self.conv1x1_3(x3)
        x4 = self.conv1x1_4(x4)

        
        x2 = self.upconv[0](x2) #28
        x3 = self.upconv[1](x3) #14
        x4 = self.upconv[2](x4) #14

        # x1 = F.interpolate(x1, x0.size()[2:], mode='bilinear', align_corners=False) #28->56
        x2 = F.interpolate(x2, x1.size()[2:], mode='bilinear', align_corners=False) #14->56
        x3 = F.interpolate(x3, x1.size()[2:], mode='bilinear', align_corners=False) 
        x4 = F.interpolate(x4, x1.size()[2:], mode='bilinear', align_corners=False) 

     

        x, gate_avg = self.fusion([x1, x2, x3, x4])
    
        x = self.ex_attn(x, gate_maps=gate_avg)

        x = self.last_conv1x1(x)

        x =  F.interpolate(x, image_shape, mode='bilinear', align_corners=False)
        
        return x



class GateFusionModule(nn.Module):
    def __init__(self, in_channels_list=[64, 64, 64, 64], out_channels=64):
        super(GateFusionModule, self).__init__()
        # Generate a separate gating map for each feature map
        self.gate_convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list])
        self.sigmoid = nn.Sigmoid()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, features):
        # features: list of feature maps from different levels
        # Example: [x1, x2, x3, x4]
        
        # Generate gate maps for each feature map
        gates = [self.sigmoid(conv(f)) for f, conv in zip(features, self.gate_convs)]
        
        fused_features = []
        for i, f in enumerate(features):
            gate_i = gates[i]
            other_gates = [g for j, g in enumerate(gates) if j != i]
            other_features = [cf for j, cf in enumerate(features) if j != i]
            
            fused_f = gate_i * f + (1 - gate_i) * sum(g * cf for g, cf in zip(other_gates, other_features))
            fused_f = f * fused_f
            fused_features.append(fused_f)
        
        fused_features = torch.cat(fused_features, dim=1)
        final_fused_feature = self.fuse_conv(fused_features)
        
        gate_avg = torch.stack(gates).mean(dim=0)

        return final_fused_feature, gate_avg
        

class ExternalAttention(nn.Module):
    def __init__(self, in_channels, S=64, memory_size=16):
        super(ExternalAttention, self).__init__()
        self.in_channels = in_channels
        self.S = S
        self.memory_size = memory_size

        self.conv1 = nn.Conv2d(in_channels, S, kernel_size=1)
        self.conv2 = nn.Conv2d(S, in_channels, kernel_size=1)

        self.linear_e = nn.Linear(S, S, bias=False)
        self.linear_f = nn.Linear(S, S, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.memory_bank = None  

    def forward(self, x, gate_maps=None):
        idx = x
        B, C, H, W = x.shape
        N = H * W

        x_proj = self.conv1(x)  # [B, S, H, W]
        x_flat = x_proj.view(B, self.S, -1).permute(0, 2, 1)  # [B, N, S]

        attn = self.linear_e(x_flat)  # [B, N, S]
        attn = self.softmax(attn)  # [B, N, S]
        attn = self.linear_f(attn)  # [B, N, S]

        # Fusion gate_maps
        if gate_maps is not None:
            if gate_maps.shape[2:] != (H, W):
                raise ValueError(f"gate_maps spatial shape {gate_maps.shape[2:]} != x shape {(H, W)}")
            gate_mask = gate_maps.mean(dim=1, keepdim=True)
            gate_mask = gate_mask.view(B, 1, N)  # [B, 1, N]
            attn = attn * gate_mask.permute(0, 2, 1)  # [B, N, S]

        # Update memory_bank
        self.update_memory_bank(attn.detach())  # [B, N, S]

        balance_factor = 0.4
        # # memory fusion
        if self.training and self.memory_bank is not None:
            mem_avg = self.memory_bank.mean(dim=0, keepdim=True)  # [1, N, S]
            if mem_avg.shape[1:] == attn.shape[1:]:
                # attn = (attn + mem_avg.expand_as(attn)) / 2.0
                attn = (attn * balance_factor + mem_avg.expand_as(attn) * (1 - balance_factor))
            else:
                print(f"[Warning] Memory shape {mem_avg.shape} not matched to attn shape {attn.shape}")

        out = attn.permute(0, 2, 1).contiguous().view(B, self.S, H, W)  # [B, S, H, W]
        out = self.conv2(out)  # [B, C, H, W]

        out = out + idx
        out = F.relu(out)

        return out

    def update_memory_bank(self, attn_tensor):
        # attn_tensor: [B, N, S]
        if self.memory_bank is None:
            self.memory_bank = attn_tensor
        else:
            self.memory_bank = torch.cat([self.memory_bank, attn_tensor], dim=0)  # [T, N, S]
            if self.memory_bank.shape[0] > self.memory_size:
                self.memory_bank = self.memory_bank[-self.memory_size:]  # Keep the latest memory_size items

