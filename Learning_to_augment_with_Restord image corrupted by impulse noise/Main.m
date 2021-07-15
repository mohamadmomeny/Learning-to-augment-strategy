
%% Select pretrained networks
% Warning: All of your dataset will be resized to input size of pretrained network

Main_net = resnet18; %other deep pretrained  model: shufflenet, densenet201, and so on.
global Input_size_of_Main_Network
Input_size_of_Main_Network=[224,224,3]; % set input size of main model

Child_net = alexnet; %other  shallow pretrained   model.
global Input_size_of_Child_Network
Input_size_of_Child_Network=[227,227,3]; % set input size of child model


%% Set train, validation, and test folders

Train_Folder='G:\Your_Dataset_Name\Train';
Validation_Folder='G:\Your_Dataset_Name\Validation';
Test_Folder='G:\Your_Dataset_Name\Test';

%%
% **************************
% ****    YOU CAN RUN    ***
% **************************

%% Load data
warning off;
imdsTrain = imageDatastore(Train_Folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsValidation = imageDatastore(Validation_Folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsTest = imageDatastore(Test_Folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% Resize original images to input size of MAIN network
% The pre-trained networks (AlexNet, ShuffleNet, GoogleNet, and Res-Net18) typically
% take input images of 3 channels (red, green, blue). Therefore, we use the same X-ray
% image and stack them three times to make the input image 3-channel. After making 3
% channels, we use the augmentation operations

%Resize of train folder:
for i=1:size(imdsTrain.Files)
    x=imread(imdsTrain.Files{i});
    x=imresize(x,Input_size_of_Main_Network(1:2));
    y=x;
    if min(min(min(size(x))))~=Input_size_of_Main_Network(3)
        y(:,:,2)=x;
        y(:,:,3)=x;
    end
    imwrite(y,imdsTrain.Files{i});
    
    if rem(i,100)==0
        disp('[MAIN Network] The number of images resized in the train folder is:')
        disp(i);
    end
end
disp('[MAIN Network] The number of images resized in the train folder is:')
disp(i);
disp('-------')

%Resize of validation folder:
for i=1:size(imdsValidation.Files)
    x=imread(imdsValidation.Files{i});
    x=imresize(x,Input_size_of_Main_Network(1:2));
    y=x;
    if min(min(min(size(x))))~=Input_size_of_Main_Network(3)
        y(:,:,2)=x;
        y(:,:,3)=x;
    end
    imwrite(y,imdsValidation.Files{i});
    
    if rem(i,100)==0
        disp('[MAIN Network] The number of images resized in the validation folder is:')
        disp(i);
    end
end
disp('[MAIN Network] The number of images resized in the validation folder is:')
disp(i);
disp('-------')
%Resize of test folder:
for i=1:size(imdsTest.Files)
    x=imread(imdsTest.Files{i});
    x=imresize(x,Input_size_of_Main_Network(1:2));
    y=x;
    if min(min(min(size(x))))~=Input_size_of_Main_Network(3)
        y(:,:,2)=x;
        y(:,:,3)=x;
    end
    imwrite(y,imdsTest.Files{i});
    
    if rem(i,100)==0
        disp('[MAIN Network] The number of images resized in the test folder is:')
        disp(i);
    end
end
disp('[MAIN Network] The number of images resized in the test folder is:')
disp(i);
disp('-------')

%% Resize original images to input size of CHILD network
mkdir('Child_folder\');
mkdir('Child_folder\Train\');
mkdir('Child_folder\Validation\');
mkdir('Child_folder\Test\');

%Resize of train folder:
for i=1:size(imdsTrain.Files)
    x=imread(imdsTrain.Files{i});
    x=imresize(x,Input_size_of_Child_Network(1:2));
    y=x;
    if min(min(min(size(x))))~=Input_size_of_Child_Network(3)
        y(:,:,2)=x;
        y(:,:,3)=x;
    end
    p=cd;
    s=replace(imdsTrain.Files{i},Train_Folder,'Child_folder\Train');
    mkdir(fileparts(s));
    imwrite(y,s);
    if rem(i,100)==0
        disp('[CHILD Network] The number of images resized in the train folder is:')
        disp(i);
    end
end
disp('[CHILD Network] The number of images resized in the train folder is:')
disp(i);
disp('-------')

%Resize of validation folder:
for i=1:size(imdsValidation.Files)
    x=imread(imdsValidation.Files{i});
    x=imresize(x,Input_size_of_Child_Network(1:2));
    y=x;
    if min(min(min(size(x))))~=Input_size_of_Child_Network(3)
        y(:,:,2)=x;
        y(:,:,3)=x;
    end
    p=cd;
    s=replace(imdsValidation.Files{i},Validation_Folder,'Child_folder\Validation');
    mkdir(fileparts(s));
    imwrite(y,s);
    if rem(i,100)==0
        disp('[CHILD Network] The number of images resized in the Validation folder is:')
        disp(i);
    end
end
disp('[CHILD Network] The number of images resized in the Validation folder is:')
disp(i);
disp('-------')

%Resize test folder:
for i=1:size(imdsTest.Files)
    x=imread(imdsTest.Files{i});
    x=imresize(x,Input_size_of_Child_Network(1:2));
    y=x;
    if min(min(min(size(x))))~=Input_size_of_Child_Network(3)
        y(:,:,2)=x;
        y(:,:,3)=x;
    end
    p=cd;
    s=replace(imdsTest.Files{i},Test_Folder,'Child_folder\Test');
    mkdir(fileparts(s));
    imwrite(y,s);
    if rem(i,100)==0
        disp('[CHILD Network] The number of images resized in the Test folder is:')
        disp(i);
    end
end
disp('[CHILD Network] The number of images resized in the Test folder is:')
disp(i);
disp('-------')

%% Find the best parameters
disp('Learning-to-augment strategy is running ...');
The_Best_P_Value=My_Bayesian_Optimizer(Child_net);
global The_Best_Parameter
The_Best_Parameter=The_Best_P_Value.XAtMinObjective.Parameter1;

%% Augment data with the best policy
the_best_policy_folder='Augmented_Data_with_the_best_policy\';

global im_size
im_size=Input_size_of_Main_Network(1);
while(rem(im_size,8)~=0)
    im_size=im_size+1;
end

imdsTrain.ReadSize = 10;

rng(0)

dsTrainNoisy = transform(imdsTrain,@addNoise);
dsTrain = combine(dsTrainNoisy,imdsTrain);
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
    dist=replace(imdsTrain.Files{i},Train_Folder,the_best_policy_folder);
    mkdir(fileparts(dist));    
    copyfile(source,dist);
    y=ypred(:,:,:,i) ;
    y = imresize(y,Input_size_of_Main_Network(1:2));
    imwrite(y,strcat(dist,'_augmented_.png'));
end

%% Load augmented data
imdsTrain = imageDatastore(the_best_policy_folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% Set input size of pretrained Network
Main_net.Layers(1)
inputSize = Main_net.Layers(1).InputSize;

%% Replace final layer

if isa(Main_net,'SeriesNetwork')
    lgraph = layerGraph(Main_net.Layers);
else
    lgraph = layerGraph(Main_net);
end

%% find learnable layer to replace

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]

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
    'MaxEpochs',50, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');
trainedNet_Main_rotate_flip = trainNetwork(imdsTrain,lgraph,options);

[YPredTest2,probs2]= classify(trainedNet_Main_rotate_flip,imdsTest);

%%test accuracy

accuracyTest2 = mean(YPredTest2 == imdsTest.Labels);
display(accuracyTest2)

%% Plot the confusion matrix.

figure('Units','normalized','Position',[0.2 0.2 0.4 0.3]);
cm2 = confusionchart(imdsTest.Labels,YPredTest2);
cm2.Title = 'Confusion Matrix for Test Data';
cm2.ColumnSummary = 'column-normalized';
cm2.RowSummary = 'row-normalized';


function dataOut = addNoise(data)
global The_Best_Parameter
dataOut = data;
for idx = 1:size(data,1)
    dataOut{idx} = imnoise(data{idx},'salt & pepper',The_Best_Parameter); 
end

end

function dataOut = commonPreprocessing(data)
global im_size
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


