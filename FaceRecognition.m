function  outputLabel=FaceRecognition(trainPath, testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames=ls(trainPath);
labelImgSet=folderNames(3:end,:);
trainImgSet = zeros(600,600,3,length(folderNames)-2);
for i=1:length(labelImgSet)
    imgName=ls([trainPath, labelImgSet(i, :), '\*.jpg']);
    trainImgSet(:,:,:,i) = imread([trainPath, labelImgSet(i,:), '\', imgName]);
end

%% Prepare the training image: Here we simply use the gray-scale values as template matching. 
% but you need to normalise the intensity using zero-mean method
trainTmpSet=zeros(600*600, size(trainImgSet, 4));
for i=1:size(trainImgSet, 4)
    tmpl = rgb2gray(uint8(trainImgSet(:,:,:,i)));
    tmpl = double(tmpl(:))/255';
    trainTmpSet(:, i) = (tmpl-mean(tmpl))/std(tmpl);
end



%% Match each of the test image with the stored training images and save it to outputLabel
testImgNames=ls([testPath, '*.jpg']);
outputLabel=[];
for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:)]);
    
    tmpl = rgb2gray(uint8(testImg));
    tmpl=double(tmpl(:))/255';
    tmpl=(tmpl-mean(tmpl))/std(tmpl);
    ccValue=trainTmpSet'*tmpl;
    labelIndex= find(ccValue==max(ccValue));
    outputLabel=[outputLabel; labelImgSet(labelIndex(1), :)];
end    

