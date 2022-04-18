clear
close all
%delete(findall(0));

N=1000;

imds = imageDatastore('C:\Users\pearlstl\Downloads\cards_sparse_45deg_each_2deg\', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
dsrand = shuffle(imds);
[ds_train, ds_val, ds_test] = splitEachLabel(dsrand,0.7,0.2);

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs',50, ...
    'MiniBatchSize',20, ...
    'Shuffle', 'once', ...
    'ValidationData', ds_val, ...
    'CheckpointPath', sprintf( '%s\\Checkpoints', pwd), ...
    'Plots','training-progress') ;

layers = [
    imageInputLayer([256 256 3], 'Normalization', 'zscore')
    convolution2dLayer(7, 32)
    reluLayer
    maxPooling2dLayer(16,'Stride',16)
%     convolution2dLayer(1, 32)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3, 32)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(1, 32)
%     reluLayer
%     maxPooling2dLayer(3,'Stride',2)
    fullyConnectedLayer(32, 'Name','fc1')
    reluLayer
    fullyConnectedLayer(13, 'Name','Out')
    softmaxLayer('Name','sm')
    classificationLayer('Name','classification')];

net = trainNetwork(imds,layers, options);


