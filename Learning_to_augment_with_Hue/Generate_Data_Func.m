%% Load Data
function Output=Generate_Data_Func(Parameter,Child_net)

Train_Folder='Child_folder\Train\';
Validation_Folder='Child_folder\Validation\';
Test_Folder='Child_folder\Test\';

%% Load train folder
imdsTrain = imageDatastore(Train_Folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsValidation = imageDatastore(Validation_Folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsTest = imageDatastore(Test_Folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% %% Augment data with impulse noise
temporary_folder='Child_folder\Train_with_Augmented_Data\';

for i=1:size(imdsTrain.Files)
    x=imread(imdsTrain.Files{i});
    y=jitterColorHSV(x,'hue',Parameter);
    s=replace(imdsTrain.Files{i},Train_Folder,temporary_folder);
    mkdir(fileparts(s));
    imwrite(x,s);
    imwrite(y,strcat(s,'_augmented_.png'));
end

%% Load augmented data
imdsTrain = imageDatastore(temporary_folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% Load pretrained Network
net = Child_net; 

%% Replace final layer

if isa(net,'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end

%% find learnable layer to replace

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%% freeze initial layer

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%% Train network

miniBatchSize = 15;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',25, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false);
trainedNet = trainNetwork(imdsTrain,lgraph,options);

[YPredTest2,~]= classify(trainedNet,imdsTest);
m=1-mean(YPredTest2 == imdsTest.Labels);

Output= m;




