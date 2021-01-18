% HOG + SVM
% 48.14% accuracy
function outputLabel = FaceRecognition1(trainPath, testPath)
IDS = 160; % intial downsample size
FDS = 140; % final downsample size
%% load in and preprocess all training images
folderNames=ls(trainPath);
labelImgSet=folderNames(3:end,:);
N = length(folderNames)-2;
A=zeros(FDS,FDS,N);

% for each training image
for j = 1:N
    imgName=ls([trainPath, labelImgSet(j, :), '\*.jpg']);
    u = imread([trainPath, labelImgSet(j,:), '\', imgName]);
    
    % convert to graysclae if it is rgb
    if (size(u,3) == 1) 
        G = u;
    else
        G = rgb2gray(uint8(u));
    end
    
    % resize
    D1 = imresize(G, [IDS IDS]);
    
    % find a face and crop to it
    C = croptoface(D1);
    
    %equalization to help with illumination
    E = double(histeq(uint8(C)));
    
    % second resize
    D2 = imresize(E,[FDS,FDS]);
    
    % save image
    A(:,:,j) = D2(:,:);
end
%% TRAIN:
trainedFeatures = [];
% extract the HOG features from each image
for j=1:N
    [hogFeatures, Visualization] = extractHOGFeatures(A(:,:,j));
    trainedFeatures(j, :) = hogFeatures(:);
end

% fit multi svm classifiers to the trained features
classifier = fitcecoc(trainedFeatures, labelImgSet, 'Coding', 'onevsall');
%% TEST:
testImgNames=ls([testPath, '*.jpg']);
outputLabel = char(zeros([size(testImgNames, 1), 6]));

% for each test image
for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:)]);
   
    % convert to graysclae if it is rgb
    if (size(testImg,3) == 1) 
        G = testImg;
    else
        G = rgb2gray(uint8(testImg));
    end
    
    % resize
    D1 = imresize(G, [IDS, IDS]);
    
    % find face and crop
    C = croptoface(D1);
    
    % equalization to help with illumination
    E = double(histeq(uint8(C)));
    
    % final downscale
    D2 = imresize(E,[FDS,FDS]);
    
    % extract features and use classifer to predict class
    [hogFeatures, Visualization] = extractHOGFeatures(D2);
    outputLabel(i,:) = predict(classifier, hogFeatures);
end
end