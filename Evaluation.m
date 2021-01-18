%% TOOLBOXES REQUIRED
% image_toolbox
% statistics_toolbox
% video_and_image_blockset

%%
clear all;
close all;
trainPath='.\FaceDatabase\Train\'; % provide full path here
testPath='.\FaceDatabase\Test\';
%% Baseline Method
tic;
   outputLabel=FaceRecognition(trainPath, testPath);
baseLineTime=toc

% evaluation
load testLabel 
correctP=0;
for i=1:size(testLabel,1)
    if outputLabel(i,:)==testLabel(i,:)
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%
%% Method 1 developed by you

tic;
   outputLabel1=FaceRecognition1(trainPath, testPath);
method1Time=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
 if outputLabel1(i,:)==testLabel(i,:)
     correctP=correctP+1;
 end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%
%% Method 2 developed by you
tic;
   outputLabel2=FaceRecognition2(trainPath, testPath);
method2Time=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
 if outputLabel2(i,:)==testLabel(i,:)
     correctP=correctP+1;
 end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%