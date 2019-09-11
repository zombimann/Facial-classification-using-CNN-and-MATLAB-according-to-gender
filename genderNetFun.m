%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%     This is a function to define the layers and parameters of the CNN    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [layers, options] = raceNetFun(inputSize, tbl, augmentedTestSet, ep)
layers = [
    imageInputLayer([inputSize 3])
    
    convolution2dLayer(11,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(7,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    fullyConnectedLayer(height(tbl))
    softmaxLayer
    classificationLayer];
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-2, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',ep, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedTestSet, ...
    'ValidationFrequency',10, ...
    'MiniBatchSize', 64, ...
    'Verbose',false, ...
    'Plots','training-progress');
end
 
