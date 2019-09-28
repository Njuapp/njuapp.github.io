function [avg, std] = atf_nag(streamFile, lambda_1, lambda_2)
%% hyper parameters
beta = 0.7;
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
if(isempty(data.testX))
    data.yhat = zeros((length(data.trainY)-data.chunkSize), 1);
    data.yValue = zeros((length(data.trainY)-data.chunkSize), 1);
else
    data.yhat = zeros(length(data.testY), 1);
    data.yValue = zeros(length(data.testY), 1);
end
if(isempty(data.testX))
    testY = data.trainY((data.chunkSize+1):end);
else
    testY = data.testY;
end
%% main
time = 0;
u_prev = zeros(dim,1);
u_prev_2 = zeros(dim,1);
w_prev = zeros(dim, 1);
acc_10fold = zeros(10, 1);
for k = 1: 10
  accAll = zeros(data.chunkNum - 9, 1);
  for chunkCnt = k: data.chunkNum - 10 + k
      tic;
  %     fprintf('*** Chunk %d ***\n', chunkCnt);
      dataStart = data.chunkSize * (chunkCnt - 1) + 1;
      dataEnd = data.chunkSize * chunkCnt;
      trainXCur = data.trainX(dataStart:dataEnd, :);
      trainYCur = data.trainY(dataStart:dataEnd);
      if(isempty(data.testX))
          testXCur = data.trainX((dataStart+data.chunkSize):(dataEnd+data.chunkSize), :);
      else
          testXCur = data.testX(dataStart:dataEnd, :);
      end
      [u_prev, u_prev_2, model] = modelTrain(trainXCur, trainYCur, u_prev, u_prev_2);
      time = time + toc;
      % test model
      for dataCnt = 1: data.chunkSize
          [data.yValue(dataStart+dataCnt-1), data.yhat(dataStart+dataCnt-1)] = modelClassify(testXCur(dataCnt,:));
      end
      accuracy = mean(data.yhat(dataStart: dataStart+data.chunkSize-1) == testY(dataStart: dataStart + data.chunkSize -1));
      accAll(chunkCnt) = accuracy;
  %     fprintf("Chunk accuracy: %f\n", accuracy);
  end
  acc_10fold(k) = mean(accAll);
end
avg = mean(acc_10fold);
std = var(acc_10fold);

% fprintf('Average accuracy is %.6f\n', accu);
%% functions
    function grad = derGrad(trainXCur, trainYCur, u,weight)
        grad = 2 * lambda_2 * [zeros(dim, 1); u + weight(dim+1:2*dim)] ...
            + 2 * lambda_1 * [weight(1:dim) - weight(dim+1: 2*dim); weight(dim+1:2*dim) - weight(1:dim)];
        loss_grad_tmp = - trainYCur.*(trainXCur);
        loss_grad = 2 * max(0, 1 - trainYCur .* (trainXCur * weight(1:dim))) .* loss_grad_tmp;
        grad(1:dim) = grad(1:dim) + (sum(loss_grad, 1))';
    end
    function obj = derObj(trainXCur, trainYCur, u, weight)
        loss = sum(max(0, 1 - trainYCur.*(trainXCur* weight(1:dim))).^2);
        reuse = lambda_1 * sum((weight(1:dim) - weight(dim+1:2*dim)).^2);
        smooth = lambda_2 * sum((weight(dim+1:2*dim) + u ).^2);
        obj = loss + reuse + smooth; 
    end
    function rate = armijo(trainXCur, trainYCur, u, weight, grad)
        rate = 1;
        obj_prev = derObj(trainXCur, trainYCur, u, weight);
        for i = 1: 7
            obj = derObj(trainXCur, trainYCur, u, weight - rate * grad);
            if obj_prev - obj > sum(grad.^2) * rate* 0.01
                break;
            end
            rate = rate * 0.1;
        end
    end
    function [trend, trend_before, model_new] = modelTrain(trainXCur, trainYCur, u_prev, u_prev_2)
        weight = rand(2 * dim, 1);
        u = u_prev_2 - 2 * u_prev;
        obj = derObj(trainXCur, trainYCur, u, weight);
        grad = zeros(2 * dim, 1);
        for i = 1:100000
            obj_prev = obj;
            grad = 0.7 * grad + derGrad(trainXCur, trainYCur, u, weight);
            step = armijo(trainXCur, trainYCur, u, weight, grad);
            weight = weight - step * grad;
            obj = derObj(trainXCur, trainYCur, u, weight);
            if abs(obj - obj_prev) < 1e-7
                break;
            end
%            fprintf('Current value : %.6f\n', obj);
        end
%          plot(1:i + 1, obj_v,'r', 1:i+1, loss_v,'g', 1:i+1, reuse_v, 'm',1:i+1, smooth_v,'k'); 
%          fprintf('%d iterations\n', i);
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