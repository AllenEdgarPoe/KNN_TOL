data = readtable('실험결과.xlsx');
data = data(:,2:end);
X = table2array(data);

%figure, scatter3(X(:,1), X(:,2), X(:,3), 5); title('Original dataset'), drawnow
%no_dims = round(intrinsic_dim(X, 'MLE'));
%disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
[mappedX, mapping] = compute_mapping(X, 'PCA', 3);	
figure, scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), 5); title('Result of PCA');
[mappedX, mapping] = compute_mapping(X, 'Laplacian', no_dims, 7);	
figure, scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), 5); title('Result of Laplacian Eigenmaps'); drawnow