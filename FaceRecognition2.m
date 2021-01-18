% Eigenface + Nearest Neighbour
% 31.69% accuracy
function outputLabel = FaceRecognition2(trainPath,testPath)
IDS = 80; % intial downsample size
FDS = 20; % final downsample size
%% Load and preprocess the training images
A=[];
avg = zeros(FDS^2, 1);
folderNames=ls(trainPath);
labelImgSet=folderNames(3:end,:);
N = length(folderNames)-2;
count = 0;

% for each training image
for j = 1:N
    imgName=ls([trainPath, labelImgSet(j, :), '\*.jpg']);
    u = imread([trainPath, labelImgSet(j,:), '\', imgName]);
    
    % convert to grayscale if it is rgb
    if (size(u,3) == 1) 
        G = u;
    else
        G = rgb2gray(uint8(u));
    end
    
    % resize
    D1 = imresize(G, [IDS, IDS]);

    % find a face and crop to it
    C = croptoface(D1);
    
    %equalization to help with illumination
    E = double(histeq(uint8(C)));
    
    % resize again
    D2 = imresize(E,[FDS,FDS]);
    
    % reshape pixel values to a column vector for storage
    R = reshape(D2, FDS^2, 1);

    % keep track of average image (for average face)
    A = [A, R];
    avg = avg + R;
    count = count + 1;
end
avg = avg / count;

%% Center the sample pictures at the origin
B = [];
for j = 1:N
    B(:,j) = A(:,j) - avg;
end
%% Compute the SVD
[U,S,V] = svd(B, 'econ');
V_PCA= U(:,1:N);
%% TRAIN : Project each image onto bases
PCAmodes = 1:100;
trainEigenProjs = zeros(length(PCAmodes), N);
for j=1:N
    imvec = B(:,j);
    trainEigenProjs(:,j) = imvec'*V_PCA(:,PCAmodes);
end
%% Test
testImgNames=ls([testPath, '*.jpg']);
outputLabel = char(zeros([size(testImgNames, 1), 6]));

% for each test image
for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:)]);
    
    %convert to grayscale if it is rgb
    if (size(testImg,3) == 1) 
        G = testImg;
    else
        G = rgb2gray(uint8(testImg));
    end
    
    % resize
    D1 = imresize(G, [IDS, IDS]);

    % find a face and crop to it
    C = croptoface(D1);
    
    %equalization to help with illumination
    E = double(histeq(uint8(C)));
    
    % resize again
    D2 = imresize(E,[FDS,FDS]);
    
    % reshape pixel values to a column vector
    R = reshape(D2, FDS^2, 1);
    
    % remove average face from the image
    vterm = R - avg;
    
    % project to eigenspace
    termpts = V_PCA(:,PCAmodes)'*vterm;
    
    % calculate distance from test image (in eigenspace) to each training
    % image (in eigenspace).
    [D, I] = pdist2(trainEigenProjs', termpts', 'cityblock', 'Smallest', 1);
    outputLabel(i, :)=labelImgSet(I, :);
end
end