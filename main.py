import torch
import torch.nn.functional as F
import numpy as np
import os

# 导入你现有的模型定义
from lcd.models import PatchNetAutoencoder, PointNetAutoencoder
from lcd.models import *
from lcd.losses import *
# 设置模型路径和点云库路径
model_path = '/Users/zezhang/Desktop/lcd-master/logs/LCD_new/model.pth'
point_cloud_path = '/Users/zezhang/Desktop/pc.npy'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
embedding_size = 128

# 加载 PatchNet 和 PointNet 模型
patchnet = PatchNetAutoencoder(embedding_size=embedding_size, normalize=True)
pointnet = PointNetAutoencoder(embedding_size=embedding_size, input_channels=6, output_channels=6, normalize=True)

checkpoint = torch.load(model_path, map_location=device)
patchnet.load_state_dict(checkpoint['patchnet'])
pointnet.load_state_dict(checkpoint['pointnet'])

patchnet.to(device)
pointnet.to(device)
patchnet.eval()
pointnet.eval()
print("Model loaded and set to eval mode.")

# 加载点云库
print(f"Loading point cloud from: {point_cloud_path}")
point_cloud = np.load(point_cloud_path, allow_pickle=True)
print(f"Point cloud shape: {point_cloud.shape}")

coordinates_list = [point['coordinates'] for point in point_cloud if len(point['coordinates']) == 6]
coordinates = np.array(coordinates_list)
print(f"Coordinates shape: {coordinates.shape}")


# 转换为 PyTorch 张量
point_cloud_tensor = torch.from_numpy(coordinates).float().to(device)
print(f"Point cloud tensor shape: {point_cloud_tensor.shape}")

# 输入2D点数据（假设是一个2D坐标）
input_2d_point = np.array([[-819.91254, 1053.9996]])  # 示例输入2D点坐标
print(f"Input 2D point: {input_2d_point}")
# 创建伪图像形式的张量，形状为 [1, 32, 32, 3]
input_2d_point_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
input_2d_point_tensor[0, 0, 0, :2] = torch.from_numpy(input_2d_point[0])  # 将2D点的坐标放在伪图像的第一个位置

print(f"Input 2D point tensor shape (after transform): {input_2d_point_tensor.shape}")


# 计算输入2D点的特征向量
with torch.no_grad():
    input_2d_feature = patchnet.encode(input_2d_point_tensor) 
print(f"Input 2D point feature shape: {input_2d_feature.shape}")

point_cloud = point_cloud_tensor.unsqueeze(0).transpose(1, 2)  # 转换为 [1, 3, num_points] 形状
print(f"Point cloud shape (after transform): {point_cloud.shape}")
with torch.no_grad():
    _,local_features = pointnet.encode(point_cloud)
print(f"Local features shape: {local_features.shape}")

# 初始化损失函数
chamfer_loss = ChamferLoss(input_channels=6)
triplet_loss = HardTripletLoss(margin=1.0, hardest=True)

# 计算输入2D点和点云库中每个点的损失
input_2d_feature = input_2d_feature.squeeze(0)  # 转换为 [feature_dim]
print(f"Input 2D feature shape (after squeeze): {input_2d_feature.shape}")
losses = []
num_points = 1024
num_blocks = point_cloud_tensor.shape[0] // num_points
for i in range(num_blocks):
    block = point_cloud_tensor[i*num_points:(i+1)*num_points].unsqueeze(0)  # 取出块，形状为 [1, 1024, 6]
    with torch.no_grad():
        _, local_features = pointnet.encode(block.transpose(1, 2))  # 转换为 [1, 6, 1024]
    print(f"Block {i} local features shape: {local_features.shape}")

    for j in range(local_features.shape[2]):
        point_feature = local_features[:, :, j].unsqueeze(0)  # 转换为 [1, feature_dim]
        print(f"Point {j} feature shape: {point_feature.shape}")

        # 计算Triplet损失，使用输入点特征向量作为anchor，当前点特征向量作为positive
        triplet = triplet_loss(input_2d_feature.unsqueeze(0), point_feature)
        print(f"Triplet loss for point {j} in block {i}: {triplet.item()}")  # 调试信息

        # 计算重建的点云
        with torch.no_grad():
            reconstructed_point_cloud = pointnet.decode(point_feature).unsqueeze(0)  # 转换为 [1, num_points, 6]
            print(f"Reconstructed point cloud shape for point {j} in block {i}: {reconstructed_point_cloud.shape}")

        # 计算Chamfer损失
        chamfer = chamfer_loss(block.unsqueeze(0), reconstructed_point_cloud)
        print(f"Chamfer loss for point {j} in block {i}: {chamfer.item()}")

        total_loss = chamfer + triplet
        print(f"Total loss for point {j} in block {i}: {total_loss.item()}")  # 调试信息
        losses.append((total_loss.item(), i * num_points + j))
        if j % 100 == 0:
            print(f"Processed {j+1} points in block {i}, current loss: {total_loss.item()}")

# 找到损失最小的5个点
losses.sort(key=lambda x: x[0])
top5_indices = [x[1] for x in losses[:5]]
top5_points = point_cloud.squeeze(0).transpose(1, 0)[top5_indices]

print("Top 5 points with smallest loss:")
print(top5_points.cpu().numpy())
