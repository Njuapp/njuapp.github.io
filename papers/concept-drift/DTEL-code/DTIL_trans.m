function [time,tsf_time] = DTIL_trans(streamFile, outFile, ensembleSize)
%% param
parameter.e = 0.0001;
addpath('DT/');
% ensemble
ensemble.memorySize = ensembleSize;
ensemble.memoryModel = [];              % ensemble.memoryModel(index).m
ensemble.memoryCnt = 0;
ensemble.predictModel = [];             % ensemble.predictModel(index).m
ensemble.predictScore = [];
ensemble.predictCnt = 0;
% data
data.chunkSize = 0;
data.chunkNum = 0;
data.trainX = [];
data.trainY = [];
data.testX = [];
data.testY = [];
load(streamFile);
data.yValueCnt = length(unique(data.trainY));
data.yMax = max(unique(data.trainY));
if(isempty(data.testX))
    data.yhat = zeros((length(data.trainY)-data.chunkSize), 1);
    data.yValue = zeros((length(data.trainY)-data.chunkSize), 1);
else
    data.yhat = zeros(length(data.testY), 1);
    data.yValue = zeros(length(data.testY), 1);
end
%% main
time = 0;
tsf_time = 0;
for chunkCnt = 1:data.chunkNum
    tic;
    fprintf('*** Chunk %d *** \n', chunkCnt);
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
    [~, predictNewModel] = modelTrain(trainXCur, trainYCur);
    % update memoryModel
    if(ensemble.memoryCnt < ensemble.memorySize)
        recent2memory = 0;
        ensemble.memoryCnt = ensemble.memoryCnt + 1;
        ensemble.memoryModel(ensemble.memoryCnt).m = predictNewModel;
    else
        recent2memory = insertMemoryModel(predictNewModel, trainXCur, trainYCur);
        fprintf('%d', recent2memory);
        if(recent2memory ~= 0)
            ensemble.memoryModel(recent2memory).m = predictNewModel;
        end
    end
    fprintf('memoryCnt %d\n', ensemble.memoryCnt);
    time = time + toc;
    % update predictModel
    tic;
    ensemble.predictCnt = ensemble.memoryCnt;
    for modelCnt = 1:ensemble.memoryCnt
        if(modelCnt~=recent2memory)
            ensemble.predictModel(modelCnt).m = modelTransfer(ensemble.memoryModel(modelCnt).m, trainXCur, trainYCur, 1);
        else
            ensemble.predictModel(modelCnt).m = modelTransfer(predictNewModel, trainXCur, trainYCur, 0);
        end
        ensemble.predictScore(modelCnt) = evaluateHisModelMse(ensemble.predictModel(modelCnt).m ,trainXCur, trainYCur, 0, 1);
    end
    fprintf('predictCnt %d\n', ensemble.predictCnt);
    tTmp = toc;
    time = time + tTmp;
    tsf_time = tsf_time + tTmp;
    % model test
    tic;
    for dataCnt = 1:data.chunkSize
        [data.yValue(dataStart+dataCnt-1), data.yhat(dataStart+dataCnt-1)] = modelClassify(testXCur(dataCnt, :));
    end
    time = time + toc;
end
if(isempty(data.testX))
    testY = data.trainY((data.chunkSize+1):end);
else
    testY = data.testY;
end
chunkSize = data.chunkSize;
classifyY = data.yhat;
yValue = data.yValue;
save(outFile, 'testY', 'classifyY', 'yValue', 'chunkSize', 'time', 'tsf_time');
%% functions
    function [newModel,predictNewModel] = modelTrain(x, y)
        [newModel, predictNewModel] = modelTrainDT(x, y);
    end
    function [y_value,yhat] = modelClassify(x)
        pScore = zeros(1, data.yMax);
        for pCnt = 1:ensemble.predictCnt
            [preTmp,~] = trsfPredict(ensemble.predictModel(pCnt).m, x);
            pScore(preTmp) = pScore(preTmp) + ensemble.predictScore(pCnt);
        end
        [y_value, yhat] = max(pScore);
        fprintf('%d\n', y_value); 
        y_value = y_value/sum(pScore);
        fprintf('%d\n', sum(pScore));
    end
    function modelTransfer = modelTransfer(model, x, y, tlflag)
        modelTransfer = modelTransferDT(model, x, y, tlflag);
    end
    function index = insertMemoryModel(insModel, x, y)
        memModelTmpSize = ensemble.memoryCnt + 1;
        tmpInsModel.m = insModel;
        memModelTmp = [ensemble.memoryModel, tmpInsModel];
        memCorrTmp = zeros(data.chunkSize, memModelTmpSize);
        divMatrix = zeros(memModelTmpSize, memModelTmpSize);
        % initialize the diversity matrix
        for modelCntTmp = 1:memModelTmpSize
            yTmp = predict(memModelTmp(modelCntTmp).m, x);
            memCorrTmp(:, modelCntTmp) = (yTmp == y);
            for divModCnt = 1:modelCntTmp
                if(divModCnt == modelCntTmp)
                    divMatrix(divModCnt, divModCnt) = 1;
                else
                    N11 = sum(memCorrTmp(:, divModCnt) & memCorrTmp(:, modelCntTmp));
                    N00 = sum((~memCorrTmp(:, divModCnt)) & (~memCorrTmp(:, modelCntTmp)));
                    N10 = sum(memCorrTmp(:, divModCnt) & (~memCorrTmp(:, modelCntTmp)));
                    N01 = sum((~memCorrTmp(:, divModCnt)) & memCorrTmp(:, modelCntTmp));
                    divMatrix(modelCntTmp, divModCnt) = (N11*N00 - N10*N01) / (N11*N00 + N10*N01);
                    divMatrix(divModCnt, modelCntTmp) = divMatrix(modelCntTmp, divModCnt);
                end
            end
        end
        divMatrix = 1 - divMatrix;
        % find the improper model
        divMatrixSum = sum(divMatrix);
        [~, index] = min(divMatrixSum);
        if(index > ensemble.memoryCnt)
            index = 0;
        end
    end
    function [yPre, yProb] = trsfPredict(TILModel, xPre)
        [dataNumPre, ~] = size(xPre);
        yPre = zeros(dataNumPre, 1);
        yProb = zeros(dataNumPre, 1);
        for dataCntPre = 1:dataNumPre
            [yPre(dataCntPre), yProb(dataCntPre)] = transferPredictDT(TILModel, xPre(dataCntPre,:));
        end
    end
    function errorCntArr = evaluateHisModelMse(model ,x, y, newFlag, trsFlag)
        if(newFlag == 1)
            mse = 0;
        else
            if(trsFlag == 1)
                [yhat, yprob] = trsfPredict(model, x);
            else
                [yhat, yprob, ~, ~] = predict(model, x);
                yprob = max(yprob, [], 2);
            end
            yprob(yhat~=y) = 1 - yprob(yhat~=y);
            mse = 1 - yprob;
            mse = sum(mse.*mse)/data.chunkSize;
        end
        
        mser = 0;
        for probCnt = 1:data.yMax
            prob = sum(y==probCnt)/length(y);
            mser = mser + prob * (1-prob) * (1-prob);
        end
        
        errorCntArr = 1/(mser + mse + parameter.e);
    end
end