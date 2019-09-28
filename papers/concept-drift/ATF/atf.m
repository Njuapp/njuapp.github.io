function accu = atf(streamFile, outFile)
%% hyper parameters
lambda_1 = 0.001;
lambda_2 = 0.002;
lr = 1e-5;
%data
data.chunkSize = 0;
data.chunkNum = 0;
data.trainX = [];
data.trainY = [];
data.testX = [];
data.testY = [];
load(streamFile);
dim = size(data.trainX,2) + 1;
data.trainX = [data.trainX, ones(size(data.trainX,1),1)];
if ~isempty(data.testX)
    data.testX = [data.testX, ones(size(data.testX,1),1)];
end
data.trainY(data.trainY==2) = -1;
data.trainY(data.trainY==0) = -1;
if ~isempty(data.testY)
    data.testY(data.testY==2) = -1;
    data.testY(data.testY==0) = -1;
end
if(isempty(data.testX))
    data.yhat = zeros((length(data.trainY)-data.chunkSize), 1);
    data.yValue = zeros((length(data.trainY)-data.chunkSize), 1);
else
    data.yhat = zeros(length(data.testY), 1);
    data.yValue = zeros(length(data.testY), 1);
end
%% main
time = 0;
u_prev = zeros(dim,1);
u_prev_2 = zeros(dim,1);
for chunkCnt = 1: data.chunkNum
    tic;
    fprintf('*** Chunk %d ***\n', chunkCnt);
    dataStart = data.chunkSize * (chunkCnt - 1) + 1;
    dataEnd = data.chunkSize * chunkCnt;
    trainXCur = data.trainX(dataStart:dataEnd, :);
    trainYCur = data.trainY(dataStart:dataEnd);
    if(isempty(data.testX))
        testXCur = data.trainX((dataStart+data.chunkSize):(dataEnd+data.chunkSize), :);
    else
        testXCur = data.testX(dataStart:dataEnd, :);
    end
    % model train
    [u_prev, u_prev_2, model] = modelTrain(trainXCur, trainYCur, u_prev, u_prev_2, chunkCnt);
    time = time + toc;
    % test model
    for dataCnt = 1: data.chunkSize
        [data.yValue(dataStart+dataCnt-1), data.yhat(dataStart+dataCnt-1)] = modelClassify(testXCur(dataCnt,:));
    end
end
if(isempty(data.testX))
    testY = data.trainY((data.chunkSize+1):end);
else
    testY = data.testY;
end
classifyY = data.yhat;
yValue = data.yValue;
accu = mean(classifyY == testY);
fprintf('Average accuracy is %.6f\n', accu);
save(outFile, 'testY', 'yValue', 'classifyY');
%% functions
    function [trend, trend_before, model_new] = modelTrain(trainXCur, trainYCur, u_prev, u_prev_2, chunkCnt)
        param1 = lambda_1;
        weight = rand(2*dim, 1);
        if chunkCnt <= 2
            param2 = 0;
        else
            param2 = lambda_2;
        end
        obj = sum(max(1 - trainYCur.*(trainXCur* weight(1:dim)))) + param1 * sum((weight(1:dim) - weight(dim+1:2*dim)).^2)...
            +param2 * sum(weight(dim+1:2*dim) + u_prev_2 - 2 * u_prev).^2;
        u_smooth =  2 * param2 * [zeros(dim);eye(dim)]*(u_prev_2 - 2 * u_prev);
        i = 0;
        obj_v = [obj];
        while(i < 100000)
            obj_prev = obj;
            grad = u_smooth + 2 * param1 * [eye(dim), -eye(dim); -eye(dim), eye(dim)] * weight...
                +2 * param2 * [zeros(dim), zeros(dim); zeros(dim), eye(dim)] * weight;
            loss_grad_tmp = - trainYCur.*(trainXCur * [eye(dim), zeros(dim)]);
            loss_select = ( 1 - trainYCur.*(trainXCur * weight(1:dim)) > 0);
            loss_grad_tmp(~loss_select, :) = 0;
            grad = grad + (sum(loss_grad_tmp, 1))';
            weight = weight - lr * grad;
            obj = sum(max(1 - trainYCur.*(trainXCur* weight(1:dim)), 0)) + param1 * sum((weight(1:dim) - weight(dim+1:2*dim)).^2)...
            +param2 * sum(weight(dim+1:2*dim) + u_prev_2 - 2 * u_prev).^2;
            obj_v = [obj_v, obj];
%            fprintf('Current value : %.6f\n', obj);
            i = i + 1;
            if abs(obj-obj_prev) < 1e-6
                break;
            end
        end
        plot(1:i+1, obj_v); 
        fprintf('%d iterations\n', i);
        model_new = weight;
        trend_before = u_prev;
        trend = weight(dim+1:2*dim);
    end
    function [y_value, yhat] = modelClassify(x)
        y_value = model(1:dim)'* x';
        if y_value > 0
            yhat = 1;
        else
            yhat = -1;
        end
    end
end