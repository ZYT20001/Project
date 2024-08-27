import h5py
import numpy as np

# 加载HDF5文件
file_path = '/Users/zezhang/Desktop/ipcai_2020_full_res_data.h5'
h5_file = h5py.File(file_path, 'r')

# 定义提取标记点数据的函数
def extract_landmarks(h5_group):
    landmarks = {}
    for key in h5_group.keys():
        landmarks[key] = np.array(h5_group[key]).flatten()
    return landmarks

# 定义提取体积数据的函数
def extract_volume(h5_group):
    volume_data = {
        'dir_mat': np.array(h5_group['dir-mat']),
        'origin': np.array(h5_group['origin']),
        'pixels': np.array(h5_group['pixels']),
        'spacing': np.array(h5_group['spacing'])
    }
    return volume_data

# 提取样本18-2799和18-2800的体积数据和标记点数据
sample_1_volume = extract_volume(h5_file['18-2799/vol'])
sample_2_volume = extract_volume(h5_file['18-2800/vol'])

sample_1_landmarks = extract_landmarks(h5_file['18-2799/vol-landmarks'])
sample_2_landmarks = extract_landmarks(h5_file['18-2800/vol-landmarks'])

# 构建3D点云函数，并进行下采样
def construct_point_cloud(volume_data, sample_ratio=0.01):
    origin = volume_data['origin']
    pixels = volume_data['pixels']
    spacing = volume_data['spacing']
    
    z, y, x = np.indices(pixels.shape)
    x = x * spacing[0] + origin[0]
    y = y * spacing[1] + origin[1]
    z = z * spacing[2] + origin[2]
    
    coordinates = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    
    # 添加默认颜色值（例如全黑）
    default_color = np.zeros((coordinates.shape[0], 3))
    
    # 合并坐标和颜色
    points = np.hstack((coordinates, default_color))
    
    # 下采样点云
    if sample_ratio < 1.0:
        indices = np.random.choice(len(points), size=int(len(points) * sample_ratio), replace=False)
        points = points[indices]
    
    return points

# 设置下采样比例
sample_ratio = 0.01  # 1% 的数据点

# 构建样本的3D点云，并进行下采样
point_cloud_1 = construct_point_cloud(sample_1_volume, sample_ratio=sample_ratio)
point_cloud_2 = construct_point_cloud(sample_2_volume, sample_ratio=sample_ratio)

# 打印每个样本中的点的数量
print("Number of points in sample 18-2799 after downsampling:", len(point_cloud_1))
print("Number of points in sample 18-2800 after downsampling:", len(point_cloud_2))

# 合并点云和landmarks
def combine_point_clouds_with_landmarks(point_clouds, landmarks):
    combined_points = []
    for i, point_cloud in enumerate(point_clouds):
        combined_points.extend([{'coordinates': point} for point in point_cloud])
        for coords in landmarks[i].values():
            combined_points.append({'coordinates': coords})
    return combined_points
# 合并点云和landmarks
combined_point_cloud = combine_point_clouds_with_landmarks([point_cloud_1, point_cloud_2], [sample_1_landmarks, sample_2_landmarks])

# 保存点云库和标记点数据为文件
np.save('/Users/zezhang/Desktop/combined_point_cloud_with_landmarks.npy', combined_point_cloud)

# 关闭HDF5文件
h5_file.close()

print("Combined point cloud with landmarks has been saved successfully.")
