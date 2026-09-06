import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math


def get_attribute_square(attribute_map, traj, length=17):
    """
    高效提取 patch（向量化实现，GPU-friendly）
    attribute_map: (B, H, W, C)
    traj: (B, T, 2) — 每个坐标是 [lat, long]，范围为 [0, H), [0, W)
    返回: (B, T, length, length, C)
    """
    B, H, W, C = attribute_map.shape
    T = traj.size(1)
    half_len = length // 2

    # 归一化 traj 坐标到 [-1, 1]（grid_sample 要求）
    # 注意：先 swap lat 和 long → [x, y] = [W, H]
    norm_traj = traj.clone().float()
    norm_traj[..., 0] = norm_traj[..., 0] / (H - 1) * 2 - 1  # lat → y axis
    norm_traj[..., 1] = norm_traj[..., 1] / (W - 1) * 2 - 1  # long → x axis

    # 构建一个 length x length 的相对网格（范围约为 [-r, +r]）
    rel = torch.linspace(-1, 1, steps=length, device=attribute_map.device)
    yy, xx = torch.meshgrid(rel, rel, indexing='ij')  # shape (length, length)
    base_grid = torch.stack((xx, yy), dim=-1)  # shape (length, length, 2)

    # 为每个 (B, T) 坐标复制这组网格，变为 (B, T, length, length, 2)
    grid = base_grid[None, None, ...] + norm_traj[:, :, None, None, :]  # broadcast
    grid = grid.reshape(B * T, length, length, 2)  # flatten for grid_sample

    # 重排 attribute_map → [B, C, H, W]
    fmap = attribute_map.permute(0, 3, 1, 2)  # [B, C, H, W]
    fmap = fmap.repeat_interleave(T, dim=0)  # → [B*T, C, H, W]

    # 执行采样（align_corners=True 更精确）
    sampled = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear', padding_mode='border')  # → [B*T, C, length, length]

    # reshape → [B, T, length, length, C]
    sampled = sampled.view(B, T, C, length, length).permute(0, 1, 3, 4, 2)

    return sampled  # (B, T, length, length, C)

def cx_Distance_m(cx_1, cx_2, dim):
    '''
    这里的坐标是单位为 m 的坐标
    '''
    d = 0

    for i in range(dim):
        d += (cx_1[i] - cx_2[i]) ** 2
    d = d ** 0.5

    return d

#################################################################################
#                                      MLP                                      #
#################################################################################

class visual_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, submag_length):
        super(visual_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim1 = 512
        self.hidden_dim2 = 512
        self.hidden_dim3 = 256
        self.output_dim = output_dim
        self.submag_length = submag_length

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.fc4 = nn.Linear(self.hidden_dim3, self.output_dim)
        
    def forward(self, mag, traj):
        '''
            mag: (B, 2, 64, 64)
            x: (B, T, C)
            traj: (B, T, 2)
        '''
        # 根据轨迹提取子图
        mag = mag.permute(0, 2, 3, 1)
        submag_temporal = get_attribute_square(mag, traj, self.submag_length)

        B, L, D, H, W = submag_temporal.shape # B, L, 2, 64, 64
        submag_temporal = submag_temporal.reshape(B, L, -1)  # 展平 (B, L, D*H*W)
        
        submag = torch.tanh(self.fc1(submag_temporal))
        submag = torch.tanh(self.fc2(submag))
        submag = torch.tanh(self.fc3(submag))
        submag = self.fc4(submag)  # 最后不加激活函数，保持线性输出
        
        return submag  # (B, L, mag_channels)

#################################################################################
#                                      CNN                                      #
#################################################################################

class visual_CNN(nn.Module):
    def __init__(self, args, input_dim, output_dim, submag_length):
        super().__init__()
        self.output_dim = output_dim
        self.submag_length = submag_length
        self.args = args
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.args.map_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))  # → (B*L, 512, 1, 1)
        )

        self.fc_out = nn.Linear(512, output_dim)

    def forward(self, mag, traj):
        """
        Args:
            mag:  (B, 2, 64, 64)
            traj: (B, T, 2)
        Returns:
            submag: (B, L, output_dim)
        """
        # 提取轨迹对应子图 (B, L, 2, 64, 64)
        mag = mag.permute(0, 2, 3, 1)
        submag_temporal = get_attribute_square(mag, traj, self.submag_length)

        submag_temporal = submag_temporal.permute(0, 1, 4, 2, 3)  # (B, L, C, H, W)
        B, L, C, H, W = submag_temporal.shape
        x = submag_temporal.view(B * L, C, H, W)

        # CNN 处理
        x = self.conv_layers(x)       # (B*L, 512, 1, 1)
        x = x.view(B * L, -1)         # (B*L, 512)

        # 输出向量
        x = self.fc_out(x)            # (B*L, output_dim)
        x = x.view(B, L, self.output_dim)  # (B, L, output_dim)

        return x


#################################################################################
#                                       DETR                                    #
#################################################################################

class visual_DETR(nn.Module):
    def __init__(self, args, cnn_out_dim=256, trans_dim=256, num_heads=8, num_layers=4, output_dim=128, submag_length=17):
        super(visual_DETR, self).__init__()
        self.args = args
        self.submag_length = submag_length

        map_dim = self.args.use_AL_map + self.args.use_BH_map
        # 1. CNN Backbone (e.g., ResNet18) for feature extraction
        resnet = models.resnet18(pretrained=True)
        # 修改 layer1~3 的所有 downsampling 结构为 stride=1（默认包含 stride=2）
        for layer in [resnet.layer1, resnet.layer2, resnet.layer3]:
            for block in layer:
                if hasattr(block, 'conv1'):
                    block.conv1.stride = (1, 1)
                if hasattr(block, 'downsample') and block.downsample is not None:
                    block.downsample[0].stride = (1, 1)

        # 重新定义 CNN 骨干结构，移除 maxpool
        self.backbone = nn.Sequential(
            nn.Conv2d(map_dim, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 支持2通道输入
            resnet.bn1,
            resnet.relu,
            # 去掉 maxpool（避免缩小 H, W）
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # 2. Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(self.args.batch_size, cnn_out_dim, 64, 64))  # H'=W'=8 assumed for 64x64 input

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=cnn_out_dim, nhead=num_heads, dim_feedforward=cnn_out_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Final MLP to reduce per-point feature to desired embedding dim
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, cnn_out_dim),
            nn.ReLU(),
            nn.Linear(cnn_out_dim, output_dim)
        )

    def forward(self, mag, traj):
        # mag: (B, 2, 64, 64)
        B = mag.size(0)

        # Step 0: Apply Hessian filter (Sobel operator) to enhance edge features
        assert not torch.isnan(mag).any(), "Error: 'mag' contains NaN values before applying Hessian operator."
        assert not torch.isinf(mag).any(), "Error: 'mag' contains Inf values before applying Hessian operator."
        if self.args.use_Hessian:
            mag_edges = self.hessian_operator(mag)  # Apply Hessian operator to mag
            mag_edges = torch.tanh(mag_edges) * 3.0
            
            mag = mag + mag_edges
        
        assert not torch.isnan(mag).any(), "Error: 'mag' contains NaN values after applying Hessian operator."
        assert not torch.isinf(mag).any(), "Error: 'mag' contains Inf values after applying Hessian operator."

        # Step 1: Feature extraction via CNN
        feat_map = self.backbone(mag)  # (B, C, H', W')
        _, C, H, W = feat_map.shape

        assert not torch.isnan(feat_map).any(), "Error: 'feat_map' contains NaN values."
        assert not torch.isinf(feat_map).any(), "Error: 'feat_map' contains Inf values."

        # Step 2: Add positional encoding
        feat_map = feat_map + self.pos_embed[:, :, :H, :W]  # broadcast to (B, C, H, W)

        # Step 3: Flatten for transformer input
        feat_flat = feat_map.flatten(2).permute(2, 0, 1)  # (H'*W', B, C)
        if torch.isnan(feat_flat).any() or torch.isinf(feat_flat).any():
            print("🔥 feat_flat contains NaN or Inf", feat_flat)

        trans_out = self.transformer(feat_flat)  # (H'*W', B, C)

        assert not torch.isnan(trans_out).any(), "Error: 'trans_out' contains NaN values."
        assert not torch.isinf(trans_out).any(), "Error: 'trans_out' contains Inf values."

        # Step 4: Reshape back to spatial feature map
        feat_encoded = trans_out.permute(1, 2, 0).reshape(B, C, H, W)  # (B, C, H, W)
        feat_encoded = feat_encoded.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Step 5: Align to traj -> get (B, T, length, length, C)
        submag = get_attribute_square(feat_encoded, traj, self.submag_length)  # externally defined function

        # Step 6: Reduce to (B, T, C) with MLP
        B, T, L1, L2, D = submag.shape
        submag = submag.view(B, T, -1, D)  # (B, T, L1*L2, D)
        submag = submag.mean(dim=2)  # mean pooling over spatial patch
        mag_embedded = self.mlp(submag)  # (B, T, output_dim)

        return mag_embedded
    
    def hessian_operator(self, mag):
        """
        使用Sobel算子计算图像的二阶导数（Hessian算子），增强边缘特征
        """
        # 定义 Sobel 卷积核（水平和垂直方向）
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_x = sobel_x.to(mag.device)
        sobel_y = sobel_y.to(mag.device)

        # 对每个通道分别应用 Sobel 卷积核
        grad_x = F.conv2d(mag[:, 0:1, :, :], sobel_x, padding=1)  # 针对第一个通道
        grad_y = F.conv2d(mag[:, 0:1, :, :], sobel_y, padding=1)  # 针对第一个通道

        grad_xx = F.conv2d(grad_x, sobel_x, padding=1)
        grad_yy = F.conv2d(grad_y, sobel_y, padding=1)
        grad_xy = F.conv2d(grad_x, sobel_y, padding=1)

        # 对第二个通道做同样的卷积
        if self.args.use_AL_map + self.args.use_BH_map == 2:
            grad_x_2 = F.conv2d(mag[:, 1:2, :, :], sobel_x, padding=1)  # 针对第二个通道
            grad_y_2 = F.conv2d(mag[:, 1:2, :, :], sobel_y, padding=1)  # 针对第二个通道

            grad_xx_2 = F.conv2d(grad_x_2, sobel_x, padding=1)
            grad_yy_2 = F.conv2d(grad_y_2, sobel_y, padding=1)
            grad_xy_2 = F.conv2d(grad_x_2, sobel_y, padding=1)

            # 组合每个通道的结果
            grad_xx = grad_xx + grad_xx_2
            grad_yy = grad_yy + grad_yy_2
            grad_xy = grad_xy + grad_xy_2

        # 返回二阶导数
        mag_edges = grad_xx + grad_yy

        return mag_edges