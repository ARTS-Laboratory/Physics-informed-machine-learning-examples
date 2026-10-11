clc;
clear;
close;

% load data
all_data=load('data\no_friction.mat');
all_data=all_data.array;

t=all_data(1,:,1); % time (s)
x=all_data(:,:,2); % displacement
v=all_data(:,:,3); % velocity (m/s)
a=all_data(:,:,4); % acceleration (a)
k=all_data(:,:,5); % stiffness (k)
F=all_data(:,:,6); % Force (N)


a_m=mean(mean(a)); % mean of a
a_std=std(std(a)); % standard deviation of a

k_m=mean(mean(k)); % mean of k
k_std=std(std(k)); % standard deviation of k

X=a;
y=k;

lin=(1:100);
randidx=randperm(100);
trainidx=lin(:,randidx(1:80));
testidx=lin(:,randidx(81:100));

X_train=X(trainidx,:);
X_test=X(testidx,:);

y_train = y(trainidx,:);
y_test = y(testidx,:);

X_train=zscore(X_train);
X_test=zscore(X_test);


neurons=100;

layers=[
    featureInputLayer(6000,Normalization='none')
    fullyConnectedLayer(100),
    sigmoidLayer,
    fullyConnectedLayer(100),
    sigmoidLayer,
    fullyConnectedLayer(100),
    sigmoidLayer
    fullyConnectedLayer(6000)];

net=dlnetwork(layers)

function [loss,gradients,state] = modelLoss(net,X_train,y_train)

[Y,state] = forward(net,X_train);
loss = mse(Y,y_train);
gradients = dlgradient(loss,net.Learnables);

end

numEpochs = 20;
miniBatchSize = 50;

ads = arrayDatastore(a,IterationDimension=1);
mbq = minibatchqueue(ads, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFormat="BC");

numObservationsTrain = numel(x);
numIterationsPerEpoch = 1;
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch" "LearnRate"], ...
    XLabel="Iteration");

epoch = 1;
iteration = 0;
averageGrad = [];
averageSqGrad = [];

while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    reset(mbq);
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        shuffle(mbq)
        X = next(mbq);
        y = next(mbq);
        
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,y);
        net.State = state;
        
        [net,averageGrad,averageSqGrad] = adamupdate( ...
            net,gradients,averageGrad,averageSqGrad,iteration);
        
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch);
        monitor.Progress = 100 * iteration/numIterations;
    end
end

rmse = testnet(net,X_test,y_test,"rmse")

k_pred=minibatchpredict(net,X_test);
k_pred=k_pred*k_std+k_m;

plot(t,k_pred(1,:))
hold on
plot(t,y_test(1,:),"r")
