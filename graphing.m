fileID = fopen('depthdata.txt', 'r');
data = fscanf(fileID, '%f');
fclose(fileID);

totalElements = length(data)
numColumns = 505;
numRows = length(data) / numColumns;

depthMap = reshape(data, [numColumns, numRows])';
depthMap = 1000 - 10*depthMap;

[x, y] = meshgrid(1:numColumns, 1:numRows);

figure;
surf(x, y, depthMap, 'EdgeColor', 'none');

xlabel('X');
ylabel('Y');
zlabel('Depth');
title('3D Depth Map');
colormap jet;
colorbar;

view(45, 30);