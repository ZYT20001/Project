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

def read_points_from_file(file_path):
    points = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Label") or line.startswith("Landmarks:") or line.startswith("  "):
                continue
            try:
                points.append([float(value) for value in line.split()])
            except ValueError:
                continue
    points = np.array(points)
    return points

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
    
    # 读取 FH-r 的 3D 坐标
    gt_3d_landmark = f['18-2799/vol-landmarks/ASIS-l'][:].flatten()

# 打印读取的真值坐标
print(f"True 3D coordinates: {gt_3d_landmark}")

# 从文件中读取点
file_path = '/Users/zezhang/Desktop/2799/segmentation_label_1.txt'
all_points = read_points_from_file(file_path)

print(f"Loaded all points from file for testing.")

# 加载预训练模型并进行测试
loss_results = []
for point in all_points:
    block_points_with_color = np.hstack((point.reshape(1, -1), np.ones((1, 3))))
    point_cloud_tensor = torch.tensor(block_points_with_color).float().unsqueeze(0).to(device)  # (1, 1, 6)

    with h5py.File(h5_path, 'r') as f:
        projection_pixels = f['18-2799/projections/024/image/pixels'][:]
        projection_image = Image.fromarray(projection_pixels)
        gt_landmarks = f['18-2799/projections/024/gt-landmarks/ASIS-l'][:]
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
        loss_results.append((point, loss_triplet.item()))

# 找出loss最小的点
min_loss_result = min(loss_results, key=lambda x: x[1])
print(f"Loss: {min_loss_result[1]}, Corresponding 3D coordinates: {min_loss_result[0]}")

# 获取预测的3D坐标
predicted_3d = min_loss_result[0]

# 打印预测的3D坐标
print(f"Predicted 3D coordinates: {predicted_3d}")



# 将结果写入文件
results_file_path = '/Users/zezhang/Desktop/loss_results3.txt'
with open(results_file_path, 'a') as results_file:
    results_file.write(f"2D coordinates: {landmark_2d}\n")
    results_file.write(f"True 3D coordinates: {gt_3d_landmark}\n")
    results_file.write(f"Predicted 3D coordinates: {predicted_3d}\n")
   
    results_file.write("\n") 
print(f"Results saved to {results_file_path}")
