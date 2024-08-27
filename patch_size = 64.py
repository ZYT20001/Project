# 设定patch size
patch_size = 64

# 定义损失函数
mse_loss = MSELoss()
chamfer_loss = ChamferLoss(input_channels=6)
triplet_loss = HardTripletLoss(margin=0.8, hardest=True)


# 提取patch并转换为tensor
patch = extract_patch(projection_image, landmark_2d, patch_size)
input_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
if input_tensor.shape[1] == 1:  # 如果是单通道图像
    input_tensor = input_tensor.repeat(1, 3, 1, 1)
input_tensor = input_tensor.permute(0, 2, 3, 1)  # 将输入形状变为 [1, 64, 64, 3]
with torch.no_grad():
    recon_patch, z2d = patchnet(input_tensor)
loss_2d_mse = mse_loss(recon_patch, input_tensor)
results_path = '/Users/zezhang/Desktop/loss_results.txt'
loss_results = []
for i,landmark in enumerate(points_3d_all):
        if i < 14:
           point_cloud = construct_point_cloud(volume_2799, landmark)  # 前14个用18-2799构建点云
        else:
           point_cloud = construct_point_cloud(volume_2800, landmark)  # 后14个用18-2800构建点云

        colors = np.ones((point_cloud.shape[0], 3))
        point_cloud_with_colors = np.hstack((point_cloud, colors))
        point_cloud_tensor = torch.tensor(point_cloud_with_colors).float().unsqueeze(0).to(device)  # (1, 1024, 6)
        point_cloud_tensor = point_cloud_tensor.permute(0, 1, 2)  # (1, 6, 1024)
        with torch.no_grad():
            recon_point, z3d = pointnet(point_cloud_tensor)
        if i == 3:
          landmark_info = f"{landmark} (4th landmark)"
        else:
           landmark_info = f"{landmark}"
        loss_triplet = triplet_loss(z2d, z3d)
        loss_results.append((landmark_info, loss_triplet.item()))
loss_results.sort(key=lambda x: x[1])

with open(results_path, 'w') as f_results:
    f_results.write("Projection Image 000:\n")
    for landmark_info, loss in loss_results:
        f_results.write(f"Landmark {landmark_info} - loss: {loss}\n")
        print(f"Landmark {landmark_info} - loss: {loss}")    
