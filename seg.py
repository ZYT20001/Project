import h5py
import numpy as np
import os

# 打开 h5 文件
file_path = '/Users/zezhang/Desktop/ipcai_2020_full_res_data.h5'
hf = h5py.File(file_path, 'r')

# 读取 vol-seg 信息
vol_seg_path = '18-2800/vol-seg/image/pixels'
vol_seg = hf[vol_seg_path][()]
print(f"Volume Segmentation Shape: {vol_seg.shape}")  # (y, x, z)

# 读取 vol 信息
vol_info_path = '18-2800/vol'
origin = hf[f'{vol_info_path}/origin'][:].flatten()
spacing = hf[f'{vol_info_path}/spacing'][:].flatten()
direction_matrix = hf[f'{vol_info_path}/dir-mat'][:]
pixels_shape = hf[f'{vol_info_path}/pixels'].shape

# 打印读取的数据
print(f"Origin: {origin}")
print(f"Spacing: {spacing}")
print(f"Direction Matrix: \n{direction_matrix}")
print(f"Pixels Shape: {pixels_shape}")

# 读取 vol-landmarks 信息
landmarks_path = '18-2800/vol-landmarks'
landmarks_group = hf[landmarks_path]

# 打印landmark坐标
landmark_coords = {}
for landmark_name, landmark in landmarks_group.items():
    coords = landmark[()].flatten()
    landmark_coords[landmark_name] = coords
    print(f"Landmark {landmark_name}: {coords}")

# 获取不同部分的标签
unique_labels = np.unique(vol_seg)
print(f"Unique labels in the segmentation data: {unique_labels}")

# 将体素坐标转换为3D坐标
def voxel_to_world(voxel_coords, origin, spacing, direction_matrix):
    voxel_coords = np.array(voxel_coords, dtype=float)
    world_coords = np.zeros_like(voxel_coords, dtype=float)
    world_coords[:, 0] = voxel_coords[:, 0] * spacing[0] + origin[0]
    world_coords[:, 1] = voxel_coords[:, 1] * spacing[1] + origin[1]   # y
    world_coords[:, 2] = voxel_coords[:, 2] * spacing[2] + origin[2]  # z
    return world_coords

# 检查转换函数
def world_to_voxel(world_coords, origin, spacing, direction_matrix):
    world_coords = np.array(world_coords, dtype=float)
    voxel_coords = np.zeros_like(world_coords, dtype=float)
    voxel_coords[:, 0] = (world_coords[:, 0] - origin[0]) / spacing[0] # x
    voxel_coords[:, 1] = (world_coords[:, 1] - origin[1]) / spacing[1]  # y
    voxel_coords[:, 2] = (world_coords[:, 2] - origin[2]) / spacing[2]  # z
    return voxel_coords

def find_nearest_non_zero_label(x, y, z, vol_seg):
    search_radius = 1
    while search_radius < max(vol_seg.shape):
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < vol_seg.shape[0] and 0 <= ny < vol_seg.shape[1] and 0 <= nz < vol_seg.shape[2]:
                        if vol_seg[nx, ny, nz] != 0:
                            return nx, ny, nz, vol_seg[nx, ny, nz]
        search_radius += 1
    return None, None, None, 0

# 检查landmark到segmentation的映射
# 检查landmark到segmentation的映射
landmark_to_seg = {}

landmark_to_seg = {}

for landmark_name, coords in landmark_coords.items():
    voxel_coords = world_to_voxel(coords.reshape(1, -1), origin, spacing, direction_matrix).flatten()
    print(f"Landmark {landmark_name} World Coord: {coords} -> Voxel Coord: {voxel_coords}")
    if voxel_coords.shape == (3,):
        x, y, z = np.round(voxel_coords).astype(int)
        print(f"Rounded Voxel Coordinates: x={x}, y={y}, z={z}")
        if 0 <= x < vol_seg.shape[0] and 0 <= y < vol_seg.shape[1] and 0 <= z < vol_seg.shape[2]:
            label = vol_seg[x, y, z] 
            if label == 0:
                nx, ny, nz, label = find_nearest_non_zero_label(x, y, z, vol_seg)
                if label != 0:
                    print(f"Landmark {landmark_name} moved to nearest non-zero label: {label} at x={nx}, y={ny}, z={nz}")
            if label not in landmark_to_seg:
                landmark_to_seg[label] = []
            landmark_to_seg[label].append((landmark_name, coords))
        else:
            print(f"Landmark {landmark_name} is out of bounds: x={x}, y={y}, z={z}")
    else:
        print(f"Incorrect shape for voxel_coords: {voxel_coords.shape}")

# 存储每个标签的所有3D点坐标
segmentation_points = {}

for label in unique_labels:
    if label == 0:
        continue  # 忽略背景
    coords = np.argwhere(vol_seg == label)
    world_coords = voxel_to_world(coords, origin, spacing, direction_matrix)
    segmentation_points[label] = world_coords

# 创建一个目录来存储每个标签的点云数据和landmark信息
output_dir = '/Users/zezhang/Desktop/segmentation_points_with_landmarks'
os.makedirs(output_dir, exist_ok=True)

# 将3D点坐标和landmark信息保存到文件
for label in unique_labels:
    if label == 0:
        continue
    output_file_path = os.path.join(output_dir, f'segmentation_label_{label}.txt')
    points = segmentation_points[label]
    with open(output_file_path, 'w') as f:
        f.write(f"Label {label} has {points.shape[0]} points:\n")
        np.savetxt(f, points, fmt="%.6f")
        if label in landmark_to_seg:
            f.write("Landmarks:\n")
            for landmark_name, coords in landmark_to_seg[label]:
                f.write(f"  {landmark_name}: {coords}\n")

# 输出landmark和对应的segmentation
for label, landmarks in landmark_to_seg.items():
    print(f"Label {label} contains landmarks:")
    for landmark_name, coords in landmarks:
        print(f"  {landmark_name}: {coords}")

hf.close()

print(f"Segmentation points with landmarks saved in directory: {output_dir}")
