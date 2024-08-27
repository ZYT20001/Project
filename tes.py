import os
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from lcd.losses import MSELoss, ChamferLoss, HardTripletLoss
from lcd.models import PatchNetAutoencoder, PointNetAutoencoder

from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    with open(file_path, 'r') as f:
        lines = f.readlines()
        landmarks = [list(map(float, line.strip().split())) for line in lines]
    return np.array(landmarks)

def find_nearest_points(landmarks, true_landmark, num_points=100):
    distances = np.linalg.norm(landmarks - true_landmark, axis=1)
    sorted_indices = np.argsort(distances)
    nearest_points = landmarks[sorted_indices][1:num_points+1]  # 从第2个点开始选取最近的点
    return nearest_points

def construct_point_cloud_from_txt(landmarks, center, block_size=1024):
    distances = np.linalg.norm(landmarks - center, axis=1)
    sorted_indices = np.argsort(distances)
    # 确保选取的点中不包含真实值
    selected_indices = [i for i in sorted_indices if not np.allclose(landmarks[i], center, atol=1e-8)][:block_size]
    if len(selected_indices) < block_size:
        remaining_count = block_size - len(selected_indices)
        selected_indices += np.random.choice(selected_indices, remaining_count, replace=True).tolist()
    
    block_points = landmarks[selected_indices]
    return block_points

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
        'origin': f['18-2800/vol/origin'][:],
        'spacing': f['18-2800/vol/spacing'][:],
        'pixels': f['18-2800/vol/pixels'][:]
    }
    
    # 读取 IPS-l 的 3D 坐标
    gt_3d_landmark = f['18-2800/vol-landmarks/MOF-l'][:].flatten()

# 打印读取的真值坐标
print(f"True 3D coordinates: {gt_3d_landmark}")

# 从txt文件中加载所有landmarks
landmarks_file_path = '/Users/zezhang/Desktop/2800/segmentation_label_2.txt'
all_landmarks = load_landmarks(landmarks_file_path)

# 找到最接近的100个点（不包括真实landmark本身）
nearest_points = find_nearest_points(all_landmarks, gt_3d_landmark, num_points=100)
print(f"Nearest 100 points:\n{nearest_points}")
# 构建100个点云块
point_cloud_blocks = []
for center in nearest_points:  # 对于每一个找到的最近点
    block_points = construct_point_cloud_from_txt(all_landmarks, center, block_size=1024)
    point_cloud_blocks.append((block_points, center))

print(f"Constructed {len(point_cloud_blocks)} point cloud blocks for testing.")

# 加载预训练模型并进行测试
loss_results = []
for block_points, center in point_cloud_blocks:
    block_points_with_color = np.hstack((block_points, np.ones((block_points.shape[0], 3))))
    point_cloud_tensor = torch.tensor(block_points_with_color).float().unsqueeze(0).to(device)  # (1, 1024, 6)
    
    with h5py.File(h5_path, 'r') as f:
        projection_pixels = f['18-2800/projections/005/image/pixels'][:]
        projection_image = Image.fromarray(projection_pixels)
        gt_landmarks = f['18-2800/projections/005/gt-landmarks/MOF-l'][:]
        landmark_2d = [gt_landmarks[0][0], gt_landmarks[1][0]]
        
        patch = extract_patch(projection_image, landmark_2d, patch_size)
        input_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
        if input_tensor.shape[1] == 1:  # 如果是单通道图像
            input_tensor = input_tensor.repeat(1, 3, 1, 1)
        input_tensor = input_tensor.permute(0, 2, 3, 1)  # 将输入形状变为 [1, 64, 64, 3]
        with torch.no_grad():
            recon_patch, z2d = patchnet(input_tensor)
        
        with torch.no_grad():
            recon_point, z3d = pointnet(point_cloud_tensor)

        loss_triplet = triplet_loss(z2d, z3d)
        loss_results.append((center, loss_triplet.item()))

# 打印输入点云的个数和2D坐标
print(f"Number of input point clouds: {len(point_cloud_blocks)}")
print(f"2D coordinates: {landmark_2d}")

# 打印最小的loss及对应的3D坐标
min_loss_result = min(loss_results, key=lambda x: x[1])
print(f"Minimum loss: {min_loss_result[1]}, Corresponding 3D coordinates: {min_loss_result[0]}")

# 获取预测的3D坐标
predicted_3d = min_loss_result[0]

# 打印预测的3D坐标
print(f"Predicted 3D coordinates: {predicted_3d}")

# 计算均方误差（MSE）
mse_error = mean_squared_error(gt_3d_landmark, predicted_3d)
print(f"Mean Squared Error: {mse_error}")



# 将结果写入文件
results_file_path = '/Users/zezhang/Desktop/loss_results3.txt'
with open(results_file_path, 'a') as results_file:
    results_file.write(f"2D coordinates: {landmark_2d}\n")
    results_file.write(f"True 3D coordinates: {gt_3d_landmark}\n")
    results_file.write(f"Predicted 3D coordinates: {predicted_3d}\n")
   
    results_file.write("\n") 
print(f"Results saved to {results_file_path}")
