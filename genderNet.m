%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    Main file   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear;
rootFolder = fullfile('Datasets','gender');
imds = imageDatastore(rootFolder, 'LabelSource', 'foldernames',...
    'IncludeSubfolders', true);

tbl = countEachLabel(imds);

% Because imds above contains an unequal number of images per category, let's
% first adjust it, so that the number of images in the training set is balanced.
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2});

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 1250; % minSetCount;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
tbl = countEachLabel(imds);

% resize to 110 x 110
n = 224;
ep = 100;
inputSize = [n n];
imds.ReadFcn = @(loc)myfun(imread(loc),inputSize);

numTrainFiles = 0.8;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
augmentedTrainingSet = augmentedImageDatastore(inputSize, imdsTrain);
augmentedTestSet = augmentedImageDatastore(inputSize, imdsValidation);

if ~exist('genderNet.mat', 'file')
    [layers, options] = raceNetFun(inputSize, tbl, augmentedTestSet, ep);
    net = trainNetwork(augmentedTrainingSet,layers,options);
    save('genderNet.mat','net')
else
    age_net = load('genderNet.mat');
    net = age_net.net;
end






