import h5py
import numpy as np

# 定义文件路径
file_path_1 = '/Users/zezhang/Desktop/3dmatch_output.h5'
file_path_2 = '/Users/zezhang/Desktop/3dmatch_0001.h5'

# 函数：比较HDF5数据集
def compare_h5_datasets(file_path_1, file_path_2):
    differences = {}
    
    with h5py.File(file_path_1, 'r') as f1, h5py.File(file_path_2, 'r') as f2:
        keys_1 = set(f1.keys())
        keys_2 = set(f2.keys())
        
        # 比较两个文件中共有的数据集
        common_keys = keys_1.intersection(keys_2)
        for key in common_keys:
            data1 = f1[key][:]
            data2 = f2[key][:]
            
            if not np.array_equal(data1, data2):
                differences[key] = {
                    'file_1_shape': data1.shape,
                    'file_2_shape': data2.shape,
                    'difference': True
                }
            else:
                differences[key] = {
                    'file_1_shape': data1.shape,
                    'file_2_shape': data2.shape,
                    'difference': False
                }
        
        # 找出仅在文件1中的数据集
        only_in_file_1 = keys_1 - keys_2
        for key in only_in_file_1:
            differences[key] = {
                'file_1': 'Exists',
                'file_2': 'Missing'
            }
        
        # 找出仅在文件2中的数据集
        only_in_file_2 = keys_2 - keys_1
        for key in only_in_file_2:
            differences[key] = {
                'file_1': 'Missing',
                'file_2': 'Exists'
            }
    
    return differences

# 比较提供的文件
differences = compare_h5_datasets(file_path_1, file_path_2)

# 打印差异
for key, value in differences.items():
    print(f"Dataset: {key}")
    for sub_key, sub_value in value.items():
        print(f"  {sub_key}: {sub_value}")
    print()
