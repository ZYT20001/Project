import h5py
import numpy as np

# 1. 加载数据
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        
        data = {}
        sample_count = 0
        for sample in f.keys():
            if 'vol' not in f[sample]:
                continue
            vol_data = f[sample]['vol']['pixels'][:]
            vol_landmarks = {key: f[sample]['vol-landmarks'][key][:] for key in f[sample]['vol-landmarks']}
            projections = {}
            for projection in f[sample]['projections']:
                proj_image = f[sample]['projections'][projection]['image']['pixels'][:]
                proj_landmarks = {key: f[sample]['projections'][projection]['gt-landmarks'][key][:] for key in f[sample]['projections'][projection]['gt-landmarks']}
                projections[projection] = {
                    'image': proj_image,
                    'landmarks': proj_landmarks
                }
            data[sample] = {
                'vol_data': vol_data,
                'vol_landmarks': vol_landmarks,
                'projections': projections
            }
            sample_count += 1
            if sample_count >= 4:  # 检查是否已读取四个样本
                break
    return data


def check_2d_landmarks(projection):
    img_height, img_width = projection['image'].shape
    valid_landmarks = {}
    for key, point in projection['landmarks'].items():
        x, y = point
        if 0 <= x < img_width and 0 <= y < img_height:
            valid_landmarks[key] = point
    return valid_landmarks

# 3. 提取3D点云
def extract_point_cloud(vol_data):
    cloud = []
    for z in range(vol_data.shape[2]):
        for y in range(vol_data.shape[1]):
            for x in range(vol_data.shape[0]):
                if vol_data[x, y, z] > 0:  
                    cloud.append([x, y, z])
    return np.array(cloud)

def ensure_shape(data, target_shape):
    result = np.zeros(target_shape)
    if data.ndim == 2 and len(target_shape) == 3:
        data = np.expand_dims(data, axis=-1)
        data = np.repeat(data, target_shape[-1], axis=-1)
    slices = tuple(slice(0, min(dim, target_dim)) for dim, target_dim in zip(data.shape, target_shape))
    result[slices] = data[slices]
    return result


# 5. 构建2D-3D匹配对
def build_2d_3d_pairs(data, target_shape_points, target_shape_images):
    pairs = []
    for sample, content in data.items():
        vol_data = content['vol_data']
        vol_landmarks = content['vol_landmarks']
        projections = content['projections']
        point_cloud = extract_point_cloud(vol_data)
        point_cloud = ensure_shape(point_cloud, target_shape_points[1:])
        
        for proj_key, projection in projections.items():
            valid_landmarks = check_2d_landmarks(projection)
            if valid_landmarks:
                image = ensure_shape(projection['image'], target_shape_images[1:])
                pairs.append({
                    'point_cloud': point_cloud,
                    'image': image,
                    'landmarks': valid_landmarks,
                    'vol_landmarks': vol_landmarks
                })
    return pairs

# 6. 加载官方数据集，并为补全数据做准备
def load_official_data(file_path):
    with h5py.File(file_path, 'r') as f:
        points = f['points'][:]
        images = f['images'][:]
    return points, images

def add_noise(data, noise_level=0.01):
    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

# 7. 补全数据
def complete_data(pairs, official_points, official_images, target_shape_points, target_shape_images):
    while len(pairs) < target_shape_points[0]:
        for i in range(min(target_shape_points[0] - len(pairs), len(official_points))):
            point_cloud = add_noise(official_points[i])
            point_cloud = ensure_shape(point_cloud, target_shape_points[1:])
            image = add_noise(official_images[i])
            image = ensure_shape(image, target_shape_images[1:])
            pairs.append({
                'point_cloud': point_cloud,
                'image': image,
                'landmarks': {},  # 没有landmarks
                'vol_landmarks': {}  # 没有vol_landmarks
            })
    return pairs

# 8. 保存数据
def save_pairs_h5(pairs, output_path):
    with h5py.File(output_path, 'w') as f:
        points = np.stack([pair['point_cloud'] for pair in pairs])
        images = np.stack([pair['image'] for pair in pairs])
        f.create_dataset('points', data=points, compression="gzip")
        f.create_dataset('images', data=images, compression="gzip")

# 示例使用
file_path = '/Users/zezhang/Desktop/ipcai_2020_full_res_data.h5'
official_file_path = '/Users/zezhang/Desktop/3dmatch_0001.h5'
output_path = '/Users/zezhang/Desktop/output_h5_file.h5'

# 指定目标数据结构
target_shape_points = (20000, 1024, 6)
target_shape_images = (20000, 64, 64, 3)

data = load_data(file_path)
pairs = build_2d_3d_pairs(data, target_shape_points, target_shape_images)
official_points, official_images = load_official_data(official_file_path)

# 补全数据
pairs = complete_data(pairs, official_points, official_images, target_shape_points, target_shape_images)

# 保存数据
save_pairs_h5(pairs, output_path)
