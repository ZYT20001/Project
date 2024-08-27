function estimateSingleCameraPose()
    worldPoints = [
        -4.68064,8.24861,-1124.67;
        -106.361,44.8032,-1114.35;
        38.0243,81.1414,-1035.79;
        -8.80714,40.5375,-1136.86;
        59.0661,62.9261,-1078.81;
        11.1506,57.1555,-1123.83;
    ];
    fx = -5257.732;  
    fy = -5257.732;  
    cx = 767.5;      
    cy = 767.5;      
    imageSize = [1536, 1536]; 
    intrinsics = cameraIntrinsics([abs(fx), abs(fy)], [cx, cy], imageSize);
    imagePoints = [
        596.06714,1216.2395;
        -107.01768,1011.2822;
        950.2909,699.3545;
        513.29736,1317.893;
        1043.7941,1040.243;
        644.3902,1279.842;
        
    ];
    try
        [R, t] = estimateCameraPoseUsingPnP(worldPoints, imagePoints, intrinsics, 8); 
    catch ME
        fprintf('failed: %s\n', ME.message);
        return;
    end
    fprintf('相机的旋转矩阵: \n');
    disp(R);
    fprintf('相机的平移向量: \n');
    disp(t);
end

function [R, t] = estimateCameraPoseUsingPnP(worldPoints, imagePoints, intrinsics, reprojectionErrorThreshold)
    [R, t] = estimateWorldCameraPose(imagePoints, worldPoints, intrinsics, 'MaxReprojectionError', reprojectionErrorThreshold);
end
