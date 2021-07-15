%% Load Data
function Output=Generate_Data_Func(Parameter,Child_net)

global Parameter1
Parameter1=Parameter;

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
temporary_train_folder='Child_folder\Train_with_Augmented_Data\';

global Input_size_of_Child_Network
global im_size
im_size=Input_size_of_Child_Network(1);
while(rem(im_size,8)~=0)
    im_size=im_size+1;
end

imdsTrain.ReadSize = 10;

rng(0)

dsTrainNoisy = transform(imdsTrain,@addNoise);
dsTrain  = combine(dsTrainNoisy,imdsTrain);
dsTrain = transform(dsTrain,@commonPreprocessing);

% exampleData = preview(dsTrain);
% inputs = exampleData(:,1);
% responses = exampleData(:,2);
% minibatch = cat(2,inputs,responses);
% montage(minibatch','Size',[8 2])
% title('Inputs (Left) and Responses (Right)')

imageLayer = imageInputLayer([im_size,im_size,3]);

encodingLayers = [ ...
    convolution2dLayer(3,64,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,128,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2), ...
    convolution2dLayer(3,128,'Padding','same'), ...
    reluLayer, ...
    maxPooling2dLayer(2,'Padding','same','Stride',2)];

decodingLayers = [ ...
    createUpsampleTransponseConvLayer(2,128), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,128), ...
    reluLayer, ...
    createUpsampleTransponseConvLayer(2,64), ...
    reluLayer, ...
    convolution2dLayer(3,3,'Padding','same'), ...
    clippedReluLayer(1.0), ...
    regressionLayer];

layers = [imageLayer,encodingLayers,decodingLayers];

options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',imdsTrain.ReadSize, ...
    'Verbose',false);

net = trainNetwork(dsTrain,layers,options);

ypred = predict(net,dsTrain);

for i=1:size(imdsTrain.Files)
    source=imdsTrain.Files{i};
    dist=replace(imdsTrain.Files{i},Train_Folder,temporary_train_folder);
    mkdir(fileparts(dist));    
    copyfile(source,dist);
    y=ypred(:,:,:,i) ;
    y = imresize(y,Input_size_of_Child_Network(1:2));
    imwrite(y,strcat(dist,'_augmented_.png'));
end

%% Load augmented data
imdsTrain = imageDatastore(temporary_train_folder, ...
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

function dataOut = addNoise(data)
dataOut = data;
for idx = 1:size(data,1)
    dataOut{idx} = imnoise(data{idx},'salt & pepper',Parameter1); 
end

end

function dataOut = commonPreprocessing(data)
dataOut = cell(size(data));
for col = 1:size(data,2)
    for idx = 1:size(data,1)
        temp = single(data{idx,col});
        temp = imresize(temp,[im_size,im_size]);
        temp = rescale(temp);
        dataOut{idx,col} = temp;
    end
end
end

function out = createUpsampleTransponseConvLayer(factor,numFilters)

filterSize = 2*factor - mod(factor,2);
cropping = (factor-mod(factor,2))/2;
numChannels = 1;

out = transposedConv2dLayer(filterSize,numFilters, ...
    'NumChannels',numChannels,'Stride',factor,'Cropping',cropping);
end

end
