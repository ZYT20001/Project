import os
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from lcd.losses import MSELoss, ChamferLoss, HardTripletLoss
from lcd.models import PatchNetAutoencoder, PointNetAutoencoder

from sklearn.metrics import mean_squared_error

# 设置环境变量来解决OpenMP错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# 加载预训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_size = 128
patchnet = PatchNetAutoencoder(embedding_size=embedding_size, normalize=True)
pointnet = PointNetAutoencoder(embedding_size=embedding_size, input_channels=6, output_channels=6, normalize=True)

model_path = '/Users/zezhang/Desktop/lcd-master/logs/LCD_new/model.pth'
checkpoint = torch.load(model_path, map_location=device)
patchnet.load_state_dict(checkpoint['patchnet'])
pointnet.load_state_dict(checkpoint['pointnet'])

patchnet.to(device)
pointnet.to(device)
patchnet.eval()
pointnet.eval()

def load_landmarks(file_path):
    landmarks = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                landmark = list(map(float, line.strip().split()))
                landmarks.append(landmark)
            except ValueError:
                continue
    return np.array(landmarks)

def find_nearest_points(landmarks, true_landmark, num_points=50):
    # 计算每个点到真实landmark的欧几里得距离
    distances = np.linalg.norm(landmarks - true_landmark, axis=1)
    
    # 将距离从小到大排序，并获取排序后的索引
    sorted_indices = np.argsort(distances)
    
    # 选择最近的num_points个点
    nearest_points = landmarks[sorted_indices][:num_points]
    
    # 输出最近点的距离用于检查
    for i, idx in enumerate(sorted_indices[:num_points]):
        print(f"Point {i + 1}: {landmarks[idx]}, Distance: {distances[idx]}")
    
    return nearest_points


def construct_point_cloud_block(volume_data, center, block_size=1024, radius=3):
    origin = volume_data['origin']
    pixels = volume_data['pixels']
    spacing = volume_data['spacing']
    
    z, y, x = np.indices(pixels.shape)
    x = x * spacing[0] + origin[0]
    y = y * spacing[1] + origin[1]
    z = z * spacing[2] + origin[2]

    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    close_points = points[(distances < radius) & (distances > 0)]  # 排除中心点本身
    
    if close_points.shape[0] >= block_size:
        selected_indices = np.random.choice(close_points.shape[0], block_size, replace=False)
    else:
        selected_indices = np.random.choice(close_points.shape[0], block_size, replace=True)
    
    block_points = close_points[selected_indices]
    center_point = block_points.mean(axis=0)
    
    return block_points, center_point

def extract_patch(image, point, patch_size=64):
    x, y = point
    
    half_patch_size = patch_size // 2
    
    # 计算patch的边界
    left = max(0, int(x - half_patch_size))
    upper = max(0, int(y - half_patch_size))
    right = min(image.width, int(x + half_patch_size))
    lower = min(image.height, int(y + half_patch_size))

    # 调整patch的边界，确保在图像范围内
    if right > image.width:
        right = image.width
        left = max(0, image.width - patch_size)
    if lower > image.height:
        lower = image.height
        upper = max(0, image.height - patch_size)
    
    if left >= right or upper >= lower:
        raise ValueError("Invalid patch coordinates: left >= right or upper >= lower")
    
    patch = image.crop((left, upper, right, lower))

    # 填充到64x64大小
    if patch.size != (patch_size, patch_size):
        new_patch = Image.new("RGB", (patch_size, patch_size))
        new_patch.paste(patch, (0, 0))
        patch = new_patch
    
    return patch

# 设定patch size
patch_size = 64

# 定义损失函数
mse_loss = MSELoss()
chamfer_loss = ChamferLoss(input_channels=6)
triplet_loss = HardTripletLoss(margin=0.8, hardest=True)

# 读取h5文件中的数据
h5_path = '/Users/zezhang/Desktop/ipcai_2020_full_res_data.h5'
with h5py.File(h5_path, 'r') as f:
    volume_2799 = {
        'origin': f['18-2799/vol/origin'][:],
        'spacing': f['18-2799/vol/spacing'][:],
        'pixels': f['18-2799/vol/pixels'][:]
    }
    
    # 读取 IPS-l 的 3D 坐标
    gt_3d_landmark = f['18-2799/vol-landmarks/MOF-l'][:].flatten()

# 打印读取的真值坐标
print(f"True 3D coordinates: {gt_3d_landmark}")

# 从txt文件中加载所有landmarks
landmarks_file_path = '/Users/zezhang/Desktop/2799/segmentation_label_2.txt'
all_landmarks = load_landmarks(landmarks_file_path)

# 找到最接近的10个点（不包括真实landmark本身）
nearest_points = find_nearest_points(all_landmarks, gt_3d_landmark, num_points=70)
print(f"Nearest 10 points:\n{nearest_points}")

# 加载预训练模型并进行测试
min_loss = float('inf')
best_nearest_point = None

with h5py.File(h5_path, 'r') as f:
    projection_pixels = f['18-2799/projections/007/image/pixels'][:]
    projection_image = Image.fromarray(projection_pixels)
    gt_landmarks = f['18-2799/projections/007/gt-landmarks/MOF-l'][:]
    landmark_2d = [gt_landmarks[0][0], gt_landmarks[1][0]]
    
    patch = extract_patch(projection_image, landmark_2d, patch_size)
    input_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
    if input_tensor.shape[1] == 1:  # 如果是单通道图像
        input_tensor = input_tensor.repeat(1, 3, 1, 1)
    input_tensor = input_tensor.permute(0, 2, 3, 1)  # 将输入形状变为 [1, 64, 64, 3]
    with torch.no_grad():
        recon_patch, z2d = patchnet(input_tensor)
    
    for nearest_point in nearest_points:
        # 围绕最近点构建点云
        block_points, _ = construct_point_cloud_block(volume_2799, nearest_point, block_size=1024)
        block_points_with_color = np.hstack((block_points, np.ones((block_points.shape[0], 3))))
        point_cloud_tensor = torch.tensor(block_points_with_color).float().unsqueeze(0).to(device)  # (1, 1024, 6)

        # 进行点云特征提取
        with torch.no_grad():
            recon_point, z3d = pointnet(point_cloud_tensor)

        # 计算损失
        loss_triplet = triplet_loss(z2d, z3d)

        if loss_triplet.item() < min_loss:
            min_loss = loss_triplet.item()
            best_nearest_point = nearest_point

# 打印结果
print(f"Best nearest point: {best_nearest_point}")
print(f"Minimum Triplet Loss: {min_loss}")

# 计算均方误差（MSE）
mse_error = mean_squared_error(gt_3d_landmark, best_nearest_point)
print(f"Mean Squared Error: {mse_error}")

# 将结果写入文件
results_file_path = '/Users/zezhang/Desktop/loss_results3.txt'
with open(results_file_path, 'a') as results_file:
    results_file.write(f"2D coordinates: {landmark_2d}\n")
    results_file.write(f"Best point: {best_nearest_point}\n")
    results_file.write(f"Mean Squared Error: {mse_error}\n")
    results_file.write("\n") 
print(f"Results saved to {results_file_path}")
